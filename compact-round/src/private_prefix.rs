use ark_bn254::Fr;
use ark_ff::{PrimeField, UniformRand};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
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
use crate::{decode, prove, setup, verify, write};

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
struct Witness {
    inputs: Vec<Fr>,
    public: [Fr; 2],
    order_salt: Fr,
    output_salt: Fr,
}

#[derive(Clone)]
struct PrivatePrefix {
    template: Template,
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
        enforce_prefix(&numeric, &self.template.shape)?;

        let config = parameters();
        let committed_order = [
            Var::Constant(Fr::from(ORDER_TAG)),
            Var::Constant(self.template.pin[0]),
            Var::Constant(self.template.pin[1]),
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

pub(crate) fn run(command: &str, args: &[String]) -> Result<Value> {
    match (command, args) {
        ("private-prefix-shape", [template]) => {
            let cs = ConstraintSystem::new_ref();
            cs.set_mode(SynthesisMode::Setup);
            PrivatePrefix {
                template: Template::load(template)?,
                witness: None,
            }
            .generate_constraints(cs.clone())?;
            Ok(
                json!({"constraints": cs.num_constraints(), "public_inputs": cs.num_instance_variables() - 1,
                      "witnesses": cs.num_witness_variables()}),
            )
        }
        ("private-prefix-setup", [template, dir]) => setup(
            dir,
            PrivatePrefix {
                template: Template::load(template)?,
                witness: None,
            },
        ),
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
        (
            "private-prefix-prove",
            [template, manifest, pin, adapter, order_salt, output_salt, commitments, key, proof],
        ) => {
            let template = Template::load(template)?;
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
                template,
                witness: Some(Witness {
                    inputs,
                    public,
                    order_salt,
                    output_salt,
                }),
            };
            let cs = ConstraintSystem::new_ref();
            circuit.clone().generate_constraints(cs.clone())?;
            if !cs.is_satisfied()? {
                return Err("private prefix circuit rejected the release statement".into());
            }
            drop(cs);
            prove(circuit, key, proof)
        }
        ("private-prefix-verify", [commitments, order_pin, vk, pin, proof]) => {
            let public = pinned_commitments(commitments, order_pin)?;
            verify(&public, vk, pin, proof)
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
            "private-prefix-open-output ADAPTER OUTPUT_SALT COMMITMENTS ORDER_PIN"
        ]})),
        _ => Err("use private-prefix-help for private prefix command syntax".into()),
    }
}
