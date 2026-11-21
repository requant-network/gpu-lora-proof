use ark_bn254::{Bn254, Fr};
use ark_groth16::Proof;
use ark_ff::{PrimeField, UniformRand};
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::{alloc::AllocVar, boolean::Boolean};
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError, SynthesisMode,
};
use rand::rngs::OsRng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::prefix::{enforce_prefix, nonnegative, Shape};
use crate::private::{
    commitment, commitment_var, constrain_order, owner_directory, parameters, pinned_commitments,
    ORDER_FIELDS,
};
use crate::statement::{read, sha256, Adapter, Manifest, Result, Statement, PUBLIC};
use crate::{check_proof, create_proof, decode, load_proving_key, prepare_key, prove, setup, verify, write};

const ORDER_TAG: u64 = u64::from_be_bytes(*b"lora-in2");
const OUTPUT_TAG: u64 = u64::from_be_bytes(*b"loraout2");
type Var = FpVar<Fr>;
type CircuitResult<T> = std::result::Result<T, SynthesisError>;

#[derive(Clone)]
struct Template {
    shape: Shape,
    pin: [Fr; 2],
}

impl Template {
    fn load(path: &str) -> Result<Self> {
        let bytes = read(path, 8 << 10)?;
        let shape = Shape::from_bytes(&bytes)?;
        let digest = Sha256::digest(&bytes);
        Ok(Self {
            shape,
            pin: [
                Fr::from_le_bytes_mod_order(&digest[..16]),
                Fr::from_le_bytes_mod_order(&digest[16..]),
            ],
        })
    }

    fn order_header(&self, ceiling: Fr, salt: Fr) -> [Fr; 5] {
        [Fr::from(ORDER_TAG), self.pin[0], self.pin[1], ceiling, salt]
    }
}

#[derive(Clone)]
struct Branch {
    template: Template,
    decoy: Vec<Fr>,
}

impl Branch {
    fn load(paths: &[String]) -> Result<Self> {
        let [shape, manifest, pin, adapter] = paths else {
            return Err("expected a shape and its pinned reference order and adapter".into());
        };
        Ok(Self {
            template: Template::load(shape)?,
            decoy: Statement::load_prefix(manifest, pin, adapter)?.prefix_inputs().split_off(2),
        })
    }

    fn constrain(&self, numeric: &[Var], active: &Boolean<Fr>) -> CircuitResult<()> {
        let routed = numeric.iter().zip(&self.decoy)
            .map(|(value, decoy)| active.select(value, &Var::Constant(*decoy)))
            .collect::<CircuitResult<Vec<_>>>()?;
        enforce_prefix(&routed, &self.template.shape)
    }
}

#[derive(Clone)]
enum Policy {
    Single(Template),
    Pair([Branch; 2]),
    Triple([Branch; 3]),
}

impl Policy {
    fn pair(paths: &[String]) -> Result<Self> {
        if paths.len() != 8 {
            return Err("expected two pinned prefix fixtures".into());
        }
        let first = Branch::load(&paths[..4])?;
        let second = Branch::load(&paths[4..])?;
        if first.template.pin == second.template.pin {
            return Err("pair requires two distinct prefix templates".into());
        }
        Ok(Self::Pair([first, second]))
    }

    fn triple(paths: &[String]) -> Result<Self> {
        if paths.len() != 12 {
            return Err("expected three pinned prefix fixtures".into());
        }
        let first = Branch::load(&paths[..4])?;
        let second = Branch::load(&paths[4..8])?;
        let third = Branch::load(&paths[8..])?;
        if first.template.pin == second.template.pin
            || first.template.pin == third.template.pin
            || second.template.pin == third.template.pin
        {
            return Err("triple requires three distinct prefix templates".into());
        }
        Ok(Self::Triple([first, second, third]))
    }

    fn template(&self, choice: usize) -> Result<&Template> {
        match self {
            Self::Single(template) => Ok(template),
            Self::Pair(branches) => branches
                .get(choice)
                .map(|branch| &branch.template)
                .ok_or_else(|| "pair choice must be 0 or 1".into()),
            Self::Triple(branches) => branches
                .get(choice)
                .map(|branch| &branch.template)
                .ok_or_else(|| "triple choice must be 0, 1 or 2".into()),
        }
    }

    fn constrain(
        &self,
        numeric: &[Var],
        cs: ConstraintSystemRef<Fr>,
        choice: Option<usize>,
    ) -> CircuitResult<[Var; 2]> {
        match self {
            Self::Single(template) => {
                enforce_prefix(numeric, &template.shape)?;
                Ok(template.pin.map(Var::Constant))
            }
            Self::Pair([first, second]) => {
                let pick_second = Boolean::new_witness(cs, || {
                    Ok(choice.ok_or(SynthesisError::AssignmentMissing)? == 1)
                })?;
                // Both branches are enforced. The inactive one receives its public valid fixture.
                for (branch, active) in [(first, !&pick_second), (second, pick_second.clone())] {
                    branch.constrain(numeric, &active)?;
                }
                let [a, b] = first.template.pin.map(Var::Constant);
                let [c, d] = second.template.pin.map(Var::Constant);
                Ok([pick_second.select(&c, &a)?, pick_second.select(&d, &b)?])
            }
            Self::Triple(branches) => {
                let zero = Var::Constant(Fr::from(0));
                let mut selector_sum = zero.clone();
                let mut pin = [zero.clone(), zero];
                for (index, branch) in branches.iter().enumerate() {
                    let selected = Boolean::new_witness(cs.clone(), || {
                        Ok(choice.ok_or(SynthesisError::AssignmentMissing)? == index)
                    })?;
                    branch.constrain(numeric, &selected)?;
                    let selected = Var::from(selected);
                    selector_sum += &selected;
                    for (value, limb) in pin.iter_mut().zip(branch.template.pin) {
                        *value += &selected * limb;
                    }
                }
                selector_sum.enforce_equal(&Var::Constant(Fr::from(1)))?;
                Ok(pin)
            }
        }
    }
}

#[derive(Clone)]
struct Witness {
    choice: usize,
    inputs: Vec<Fr>,
    public: [Fr; 2],
    order_salt: Fr,
    output_salt: Fr,
}

#[derive(Clone)]
struct PrivatePrefix {
    policy: Policy,
    witness: Option<Witness>,
}

impl ConstraintSynthesizer<Fr> for PrivatePrefix {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> CircuitResult<()> {
        let witness = || {
            self.witness
                .as_ref()
                .ok_or(SynthesisError::AssignmentMissing)
        };
        let order = Var::new_input(cs.clone(), || Ok(witness()?.public[0]))?;
        let output = Var::new_input(cs.clone(), || Ok(witness()?.public[1]))?;
        let order_salt = Var::new_witness(cs.clone(), || Ok(witness()?.order_salt))?;
        let output_salt = Var::new_witness(cs.clone(), || Ok(witness()?.output_salt))?;
        let numeric = (0..PUBLIC - 1)
            .map(|i| Var::new_witness(cs.clone(), || Ok(witness()?.inputs[i])))
            .collect::<CircuitResult<Vec<_>>>()?;
        let (order_fields, output_and_ceiling) = numeric.split_at(ORDER_FIELDS);
        let (ceiling, adapter) = output_and_ceiling.split_last().unwrap();
        constrain_order(order_fields)?;
        nonnegative(ceiling, 64)?;
        let [first_pin, second_pin] = self.policy.constrain(
            &numeric,
            cs.clone(),
            self.witness.as_ref().map(|w| w.choice),
        )?;

        let config = parameters();
        let committed_order = [
            Var::Constant(Fr::from(ORDER_TAG)),
            first_pin,
            second_pin,
            ceiling.clone(),
            order_salt,
        ];
        commitment_var(cs.clone(), &committed_order, order_fields, &config)?
            .enforce_equal(&order)?;
        let committed_output = [Var::Constant(Fr::from(OUTPUT_TAG)), order, output_salt];
        commitment_var(cs, &committed_output, adapter, &config)?.enforce_equal(&output)?;
        Ok(())
    }
}

fn output_commitment(order: Fr, adapter: &[Fr], salt: Fr) -> Fr {
    commitment(&[Fr::from(OUTPUT_TAG), order, salt], adapter, &parameters())
}

fn shape(policy: Policy) -> Result<Value> {
    let cs = ConstraintSystem::new_ref();
    cs.set_mode(SynthesisMode::Setup);
    PrivatePrefix {
        policy,
        witness: None,
    }
    .generate_constraints(cs.clone())?;
    Ok(
        json!({ "constraints": cs.num_constraints(), "public_inputs": cs.num_instance_variables() - 1,
              "witnesses": cs.num_witness_variables() }),
    )
}

fn checked_order(policy: Policy, choice: usize, args: &[String]) -> Result<PrivatePrefix> {
    let [manifest, pin, adapter, order_salt, output_salt, commitments] = args else {
        return Err("expected manifest, adapter, openings and commitments".into());
    };
    let template = policy.template(choice)?;
    let inputs = Statement::load_prefix(manifest, pin, adapter)?
        .prefix_inputs()
        .split_off(2);
    let public: [Fr; 2] = decode(&read(commitments, 64)?)?;
    let order_salt = decode(&read(order_salt, 32)?)?;
    let output_salt = decode(&read(output_salt, 32)?)?;
    let header = template.order_header(*inputs.last().unwrap(), order_salt);
    if commitment(&header, &inputs[..ORDER_FIELDS], &parameters()) != public[0]
        || output_commitment(
            public[0],
            &inputs[ORDER_FIELDS..inputs.len() - 1],
            output_salt,
        ) != public[1]
    {
        return Err("private prefix opening does not match the commitments".into());
    }
    let circuit = PrivatePrefix {
        policy,
        witness: Some(Witness {
            inputs,
            public,
            order_salt,
            output_salt,
            choice,
        }),
    };
    let cs = ConstraintSystem::new_ref();
    circuit.clone().generate_constraints(cs.clone())?;
    if !cs.is_satisfied()? {
        return Err("private prefix circuit rejected the release statement".into());
    }
    Ok(circuit)
}

fn prove_prefix(policy: Policy, choice: usize, args: &[String]) -> Result<Value> {
    let [order @ .., key, proof] = args else {
        return Err("expected prefix order, key and proof paths".into());
    };
    prove(checked_order(policy, choice, order)?, key, proof)
}

fn prove_batch<const N: usize>(
    key: &str, output: &str, policy: Policy, orders: &[String],
) -> Result<Value> {
    let circuits = orders.chunks_exact(7).map(|order| {
        checked_order(policy.clone(), order[0].parse()?, &order[1..])
    }).collect::<Result<Vec<_>>>()?;
    let (key, key_load_seconds) = load_proving_key(key)?;
    let mut proofs = Vec::with_capacity(N);
    let mut proof_seconds = Vec::with_capacity(N);
    for circuit in circuits {
        let (proof, elapsed) = create_proof(circuit, &key)?;
        proofs.push(proof);
        proof_seconds.push(elapsed);
    }
    let proofs: [Proof<Bn254>; N] = proofs.try_into().map_err(|_| "batch proof count mismatch")?;
    write(output, &proofs)?;
    Ok(json!({
        "proof_count": N, "proof_bytes": std::fs::metadata(output)?.len(),
        "key_load_seconds": key_load_seconds, "proof_seconds": proof_seconds,
    }))
}

fn verify_batch(vk: &str, pin: &str, orders: &[String]) -> Result<Value> {
    let commitments = orders.chunks_exact(3)
        .map(|order| pinned_commitments(&order[0], &order[1]))
        .collect::<Result<Vec<_>>>()?;
    let prepared = prepare_key(vk, pin)?;
    for (public, order) in commitments.iter().zip(orders.chunks_exact(3)) {
        let proof = decode(&read(&order[2], 1024)?)?;
        check_proof(public, &prepared, &proof)?;
    }
    Ok(json!({"accepted": true, "proof_count": commitments.len()}))
}

pub(crate) fn run(command: &str, args: &[String]) -> Result<Value> {
    match (command, args) {
        ("private-prefix-shape", [template]) => shape(Policy::Single(Template::load(template)?)),
        ("private-prefix-pair-shape", paths) => shape(Policy::pair(paths)?),
        ("private-prefix-triple-shape", paths) => shape(Policy::triple(paths)?),
        ("private-prefix-setup", [template, dir]) => setup(
            dir,
            PrivatePrefix {
                policy: Policy::Single(Template::load(template)?),
                witness: None,
            },
        ),
        ("private-prefix-pair-setup", [dir, paths @ ..]) => setup(
            dir,
            PrivatePrefix {
                policy: Policy::pair(paths)?,
                witness: None,
            },
        ),
        ("private-prefix-triple-setup", [dir, paths @ ..]) => setup(
            dir,
            PrivatePrefix {
                policy: Policy::triple(paths)?,
                witness: None,
            },
        ),
        ("private-prefix-triple-prove", [choice, paths @ ..]) if paths.len() == 20 => {
            let (policy, proof) = paths.split_at(12);
            prove_prefix(Policy::triple(policy)?, choice.parse()?, proof)
        }
        ("private-prefix-pair-prove", [choice, paths @ ..]) if paths.len() == 16 => {
            let (policy, proof) = paths.split_at(8);
            prove_prefix(Policy::pair(policy)?, choice.parse()?, proof)
        }
        ("private-prefix-pair-prove-two", [key, output, paths @ ..]) if paths.len() == 22 => {
            let (policy, orders) = paths.split_at(8);
            prove_batch::<2>(key, output, Policy::pair(policy)?, orders)
        }
        ("private-prefix-triple-prove-three", [key, output, paths @ ..]) if paths.len() == 33 => {
            let (policy, orders) = paths.split_at(12);
            prove_batch::<3>(key, output, Policy::triple(policy)?, orders)
        }
        ("private-prefix-order", [template, manifest, pin, dir]) => {
            let template = Template::load(template)?;
            let (manifest, _) = Manifest::load_prefix(manifest, pin)?;
            let salt = Fr::rand(&mut OsRng);
            let header = template.order_header(manifest.heldout_ceiling(), salt);
            let order = commitment(&header, &manifest.order_fields(), &parameters());
            let dir = owner_directory(dir)?;
            write(dir.join("order.salt"), &salt)?;
            let file = dir.join("order.commitment");
            write(&file, &order)?;
            Ok(json!({"order_commitment_sha256": sha256(&read(file, 32)?), "commitment_bytes": 32}))
        }
        ("private-prefix-output", [order, adapter, dir]) => {
            let order = decode(&read(order, 32)?)?;
            let adapter: Vec<_> = Adapter::load(adapter)?.values().collect();
            let salt = Fr::rand(&mut OsRng);
            let output = output_commitment(order, &adapter, salt);
            let dir = owner_directory(dir)?;
            write(dir.join("output.salt"), &salt)?;
            write(dir.join("round.commitments"), &[order, output])?;
            Ok(json!({"public_statement_bytes": 64}))
        }
        ("private-prefix-prove", [template, proof @ ..]) => {
            prove_prefix(Policy::Single(Template::load(template)?), 0, proof)
        }
        ("private-prefix-verify", [commitments, order_pin, vk, pin, proof]) => {
            let public = pinned_commitments(commitments, order_pin)?;
            verify(&public, vk, pin, proof)
        }
        ("private-prefix-verify-two", [vk, pin, orders @ ..]) if orders.len() == 6 => {
            verify_batch(vk, pin, orders)
        }
        ("private-prefix-verify-three", [vk, pin, orders @ ..]) if orders.len() == 9 => {
            verify_batch(vk, pin, orders)
        }
        ("private-prefix-open-output", [adapter, output_salt, commitments, order_pin]) => {
            let public = pinned_commitments(commitments, order_pin)?;
            let adapter: Vec<_> = Adapter::load(adapter)?.values().collect();
            let salt = decode(&read(output_salt, 32)?)?;
            if output_commitment(public[0], &adapter, salt) != public[1] {
                return Err("private prefix adapter opening mismatch".into());
            }
            Ok(json!({"accepted": true}))
        }
        ("private-prefix-help", []) => Ok(json!({"commands": [
            "private-prefix-shape SHAPE", "private-prefix-setup SHAPE DIR",
            "private-prefix-order SHAPE MANIFEST PIN DIR", "private-prefix-output ORDER ADAPTER DIR",
            "private-prefix-prove SHAPE MANIFEST PIN ADAPTER ORDER_SALT OUTPUT_SALT COMMITMENTS PK PROOF",
            "private-prefix-verify COMMITMENTS ORDER_PIN VK VK_PIN PROOF",
            "private-prefix-verify-two VK VK_PIN FIRST_COMMITMENTS FIRST_ORDER_PIN FIRST_PROOF SECOND_COMMITMENTS SECOND_ORDER_PIN SECOND_PROOF",
            "private-prefix-verify-three VK VK_PIN ORDER_PROOF ORDER_PROOF ORDER_PROOF",
            "ORDER_PROOF = COMMITMENTS ORDER_PIN PROOF",
            "private-prefix-open-output ADAPTER OUTPUT_SALT COMMITMENTS ORDER_PIN",
            "private-prefix-pair-shape FIXTURES",
            "private-prefix-pair-setup DIR FIXTURES",
            "private-prefix-pair-prove CHOICE FIXTURES MANIFEST PIN ADAPTER ORDER_SALT OUTPUT_SALT COMMITMENTS PK PROOF",
            "private-prefix-pair-prove-two PK OUTPUT FIXTURES ORDER ORDER",
            "private-prefix-triple-shape FIXTURE FIXTURE FIXTURE",
            "private-prefix-triple-setup DIR FIXTURE FIXTURE FIXTURE",
            "private-prefix-triple-prove CHOICE FIXTURE FIXTURE FIXTURE MANIFEST PIN ADAPTER ORDER_SALT OUTPUT_SALT COMMITMENTS PK PROOF",
            "private-prefix-triple-prove-three PK OUTPUT FIXTURE FIXTURE FIXTURE ORDER ORDER ORDER",
            "FIXTURE = SHAPE MANIFEST PIN ADAPTER",
            "ORDER = CHOICE MANIFEST PIN ADAPTER ORDER_SALT OUTPUT_SALT COMMITMENTS",
            "FIXTURES = FIRST_SHAPE FIRST_MANIFEST FIRST_PIN FIRST_ADAPTER SECOND_SHAPE SECOND_MANIFEST SECOND_PIN SECOND_ADAPTER"
        ]})),
        _ => Err("use private-prefix-help for private prefix command syntax".into()),
    }
}
