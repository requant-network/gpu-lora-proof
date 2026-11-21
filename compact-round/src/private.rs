use ark_bn254::{Bn254, Fr};
use ark_crypto_primitives::sponge::constraints::CryptographicSpongeVar;
use ark_crypto_primitives::sponge::poseidon::{
    constraints::PoseidonSpongeVar, find_poseidon_ark_and_mds, PoseidonConfig, PoseidonSponge,
};
use ark_crypto_primitives::sponge::{CryptographicSponge, FieldBasedCryptographicSponge};
use ark_ff::UniformRand;
use ark_groth16::{prepare_verifying_key, Groth16, Proof, ProvingKey, VerifyingKey};
use ark_r1cs_std::{alloc::AllocVar, boolean::Boolean, eq::EqGadget, fields::fp::FpVar};
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError, SynthesisMode,
};
use rand::rngs::OsRng;
use serde_json::{json, Value};
use std::fs::{self, DirBuilder};
use std::path::Path;
use std::time::Instant;

use crate::circuit::{enforce_round, magnitude};
use crate::statement::{
    read, sha256, Adapter, Manifest, Result, Statement, ADAPTER, BATCH, D, HELDOUT, PUBLIC,
};
use crate::{decode, setup, write, KEY_LIMIT};

const ORDER_FIELDS: usize = PUBLIC - 2 - 2 * ADAPTER;
const PACKED_WIDTH: usize = 20;
const ORDER_TAG: u64 = u64::from_be_bytes(*b"lora-in1");
const OUTPUT_TAG: u64 = u64::from_be_bytes(*b"loraout1");
type Var = FpVar<Fr>;
type Bit = Boolean<Fr>;
type CircuitResult<T> = std::result::Result<T, SynthesisError>;

fn parameters() -> PoseidonConfig<Fr> {
    // The generated width-3 permutation matches the pinned circomlibjs table.
    let (ark, mds) = find_poseidon_ark_and_mds::<Fr>(254, 2, 8, 57, 0);
    PoseidonConfig::new(8, 57, 5, mds, ark, 2, 1)
}

fn pack(fields: &[Fr]) -> Vec<Fr> {
    fields
        .chunks(PACKED_WIDTH)
        .map(|chunk| {
            chunk.iter().rev().fold(Fr::from(0), |acc, value| {
                acc * Fr::from(4096) + value + Fr::from(1024)
            })
        })
        .collect()
}

fn pack_vars(fields: &[Var]) -> Vec<Var> {
    fields
        .chunks(PACKED_WIDTH)
        .map(|chunk| {
            chunk
                .iter()
                .rev()
                .fold(Var::Constant(Fr::from(0)), |acc, value| {
                    acc * Fr::from(4096) + value + Fr::from(1024)
                })
        })
        .collect()
}

fn commitment(prefix: &[Fr], fields: &[Fr], config: &PoseidonConfig<Fr>) -> Fr {
    let mut sponge = PoseidonSponge::new(config);
    let data: Vec<_> = prefix.iter().copied().chain(pack(fields)).collect();
    sponge.absorb(&data);
    sponge.squeeze_native_field_elements(1)[0]
}

fn commitment_var(
    cs: ConstraintSystemRef<Fr>,
    prefix: &[Var],
    fields: &[Var],
    config: &PoseidonConfig<Fr>,
) -> CircuitResult<Var> {
    let mut sponge = PoseidonSpongeVar::new(cs, config);
    let data: Vec<_> = prefix.iter().cloned().chain(pack_vars(fields)).collect();
    sponge.absorb(&data)?;
    Ok(sponge.squeeze_field_elements(1)?.remove(0))
}

fn constrain_order(fields: &[Var]) -> CircuitResult<()> {
    for (index, value) in fields.iter().enumerate() {
        let base = index < D * D;
        let adapter = index >= D * D && index < D * D + 2 * ADAPTER;
        let (_, bits, _) = magnitude(value, if base { 11 } else { 7 })?;
        // W and the initial factors may reach exactly their power-of-two limit.
        if base || adapter {
            let (top, lower) = bits.split_last().ok_or(SynthesisError::Unsatisfiable)?;
            Bit::kary_and(&[top.clone(), Bit::kary_or(lower)?])?.enforce_equal(&Bit::FALSE)?;
        }
    }
    let start = D * D + 2 * ADAPTER;
    let slot = 2 * BATCH * D;
    let training: Vec<_> = fields[start..start + 4 * slot]
        .chunks_exact(slot)
        .flat_map(|batch| batch[..BATCH * D].chunks_exact(D).map(pack_vars))
        .collect();
    let eval = start + 4 * slot;
    let heldout: Vec<_> = fields[eval..eval + HELDOUT * D]
        .chunks_exact(D)
        .map(pack_vars)
        .collect();
    for (i, row) in heldout.iter().enumerate() {
        for other in training.iter().chain(&heldout[..i]) {
            let equal = Bit::kary_and(&[row[0].is_eq(&other[0])?, row[1].is_eq(&other[1])?])?;
            equal.enforce_equal(&Bit::FALSE)?;
        }
    }
    Ok(())
}

struct PrivateWitness {
    numeric: Vec<Fr>,
    public: [Fr; 2],
    order_salt: Fr,
    output_salt: Fr,
}

struct PrivateRound(Option<PrivateWitness>);

impl ConstraintSynthesizer<Fr> for PrivateRound {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> CircuitResult<()> {
        let witness = || self.0.as_ref().ok_or(SynthesisError::AssignmentMissing);
        let order = Var::new_input(cs.clone(), || Ok(witness()?.public[0]))?;
        let output = Var::new_input(cs.clone(), || Ok(witness()?.public[1]))?;
        let order_salt = Var::new_witness(cs.clone(), || Ok(witness()?.order_salt))?;
        let output_salt = Var::new_witness(cs.clone(), || Ok(witness()?.output_salt))?;
        let numeric = (0..PUBLIC - 2)
            .map(|i| Var::new_witness(cs.clone(), || Ok(witness()?.numeric[i])))
            .collect::<CircuitResult<Vec<_>>>()?;
        let (inputs, adapter) = numeric.split_at(ORDER_FIELDS);
        constrain_order(inputs)?;
        enforce_round(&numeric)?;
        let config = parameters();
        let order_prefix = [Var::Constant(Fr::from(ORDER_TAG)), order_salt];
        commitment_var(cs.clone(), &order_prefix, inputs, &config)?.enforce_equal(&order)?;
        let output_prefix = [Var::Constant(Fr::from(OUTPUT_TAG)), order, output_salt];
        commitment_var(cs, &output_prefix, adapter, &config)?.enforce_equal(&output)?;
        Ok(())
    }
}

fn owner_directory(path: &str) -> Result<&Path> {
    let dir = Path::new(path);
    let mut builder = DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)?;
    Ok(dir)
}

fn pinned_commitments(path: &str, order_pin: &str) -> Result<[Fr; 2]> {
    let bytes = read(path, 64)?;
    let public = decode(&bytes)?;
    if sha256(&bytes[..32]) != order_pin {
        return Err("issued order commitment SHA-256 mismatch".into());
    }
    Ok(public)
}

pub(crate) fn run(command: &str, args: &[String]) -> Result<Value> {
    match (command, args) {
        ("private-shape", []) => {
            let cs = ConstraintSystem::new_ref();
            cs.set_mode(SynthesisMode::Setup);
            PrivateRound(None).generate_constraints(cs.clone())?;
            Ok(
                json!({"constraints": cs.num_constraints(), "public_inputs": cs.num_instance_variables() - 1,
                      "witnesses": cs.num_witness_variables()}),
            )
        }
        ("private-setup", [dir]) => setup(dir, PrivateRound(None)),
        ("private-order", [manifest, pin, dir]) => {
            let (manifest, _) = Manifest::load(manifest, pin)?;
            let salt = Fr::rand(&mut OsRng);
            let order = commitment(
                &[Fr::from(ORDER_TAG), salt],
                &manifest.order_fields(),
                &parameters(),
            );
            let path = owner_directory(dir)?;
            write(path.join("order.salt"), &salt)?;
            let file = path.join("order.commitment");
            write(&file, &order)?;
            Ok(json!({"commitment_bytes": 32, "order_commitment_sha256": sha256(&read(file, 32)?)}))
        }
        ("private-output", [order, adapter, dir]) => {
            let order: Fr = decode(&read(order, 32)?)?;
            let output: Vec<_> = Adapter::load(adapter)?.values().collect();
            let salt = Fr::rand(&mut OsRng);
            let result = commitment(&[Fr::from(OUTPUT_TAG), order, salt], &output, &parameters());
            let path = owner_directory(dir)?;
            write(path.join("output.salt"), &salt)?;
            write(path.join("round.commitments"), &[order, result])?;
            Ok(json!({"public_statement_bytes": 64}))
        }
        (
            "private-prove",
            [manifest, pin, adapter, order_salt, output_salt, commitments, pk, path],
        ) => {
            let statement = Statement::load(manifest, pin, adapter)?;
            let public: [Fr; 2] = decode(&read(commitments, 64)?)?;
            let order_salt: Fr = decode(&read(order_salt, 32)?)?;
            let output_salt: Fr = decode(&read(output_salt, 32)?)?;
            let numeric = statement.public_inputs().split_off(2);
            let (inputs, adapter) = numeric.split_at(ORDER_FIELDS);
            let config = parameters();
            if commitment(&[Fr::from(ORDER_TAG), order_salt], inputs, &config) != public[0]
                || commitment(
                    &[Fr::from(OUTPUT_TAG), public[0], output_salt],
                    adapter,
                    &config,
                ) != public[1]
            {
                return Err(
                    "private opening does not match the order and adapter commitments".into(),
                );
            }
            statement.replay()?;
            let key: ProvingKey<Bn254> = decode(&read(pk, KEY_LIMIT)?)?;
            let witness = PrivateWitness {
                numeric,
                public,
                order_salt,
                output_salt,
            };
            let start = Instant::now();
            let proof = Groth16::<Bn254>::create_random_proof_with_reduction(
                PrivateRound(Some(witness)),
                &key,
                &mut OsRng,
            )?;
            write(path, &proof)?;
            Ok(
                json!({"proof_bytes": fs::metadata(path)?.len(), "prove_and_write_seconds": start.elapsed().as_secs_f64()}),
            )
        }
        ("private-verify", [commitments, order_pin, vk, pin, proof]) => {
            let public = pinned_commitments(commitments, order_pin)?;
            let key = read(vk, 1 << 20)?;
            if sha256(&key) != *pin {
                return Err("verifying key SHA-256 mismatch".into());
            }
            let vk: VerifyingKey<Bn254> = decode(&key)?;
            let proof: Proof<Bn254> = decode(&read(proof, 128)?)?;
            if !Groth16::<Bn254>::verify_proof(&prepare_verifying_key(&vk), &proof, &public)? {
                return Err("Groth16 rejected the linked private commitments".into());
            }
            Ok(json!({"accepted": true, "public_inputs": public.len()}))
        }
        ("private-open-output", [adapter, output_salt, commitments, order_pin]) => {
            let public = pinned_commitments(commitments, order_pin)?;
            let output: Vec<_> = Adapter::load(adapter)?.values().collect();
            let salt: Fr = decode(&read(output_salt, 32)?)?;
            if commitment(
                &[Fr::from(OUTPUT_TAG), public[0], salt],
                &output,
                &parameters(),
            ) != public[1]
            {
                return Err("private adapter opening mismatch".into());
            }
            Ok(json!({"accepted": true}))
        }
        ("private-help", []) => Ok(json!({"commands": [
            "private-shape", "private-setup DIR", "private-order MANIFEST PIN DIR",
            "private-output ORDER_COMMITMENT ADAPTER DIR",
            "private-prove MANIFEST PIN ADAPTER ORDER_SALT OUTPUT_SALT COMMITMENTS PK PROOF",
            "private-verify COMMITMENTS ORDER_PIN VK VK_PIN PROOF", "private-open-output ADAPTER OUTPUT_SALT COMMITMENTS ORDER_PIN",
        ]})),
        _ => Err("use private-help for private-round command syntax".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_serialize::CanonicalSerialize;

    #[test]
    fn poseidon_constants_match_the_published_width_three_table() {
        let p = parameters();
        let mut bytes = Vec::new();
        for entry in p.ark.iter().chain(&p.mds).flatten() {
            entry.serialize_compressed(&mut bytes).unwrap();
        }
        assert_eq!(
            sha256(&bytes),
            "e4b7f120fccdc21a73dd32fb210837a733555cb831173a0210d606f1a7705975"
        );
    }

    #[test]
    fn wrong_commitment_leaves_the_circuit_unsatisfied() {
        let cs = ConstraintSystem::new_ref();
        let config = parameters();
        let prefix = [Fr::from(ORDER_TAG), Fr::from(42)];
        let values: Vec<_> = (-17..16).map(crate::statement::scalar).collect();
        let expected = commitment(&prefix, &values, &config);
        let wrong = commitment(&prefix, &values[..32], &config);
        assert_ne!(expected, wrong);
        let vars = values
            .iter()
            .map(|v| Var::new_witness(cs.clone(), || Ok(*v)).unwrap())
            .collect::<Vec<_>>();
        let prefix_vars = prefix.map(Var::Constant);
        let v = commitment_var(cs.clone(), &prefix_vars, &vars, &config).unwrap();
        v.enforce_equal(&Var::Constant(expected)).unwrap();
        assert!(cs.is_satisfied().unwrap());
        v.enforce_equal(&Var::Constant(wrong)).unwrap();
        assert!(!cs.is_satisfied().unwrap());
    }

    #[test]
    fn private_ranges_and_heldout_separation_are_constrained() {
        fn satisfied(values: &[i64]) -> bool {
            let cs = ConstraintSystem::new_ref();
            let vars: Vec<_> = values
                .iter()
                .map(|v| Var::new_witness(cs.clone(), || Ok(crate::statement::scalar(*v))).unwrap())
                .collect();
            constrain_order(&vars).is_ok() && cs.is_satisfied().unwrap()
        }

        let mut values = vec![0; ORDER_FIELDS];
        let training = D * D + 2 * ADAPTER;
        let heldout = training + 8 * BATCH * D;
        for i in 0..HELDOUT {
            values[heldout + i * D + i] = 1;
        }
        assert!(satisfied(&values));
        for (index, invalid) in [(0, 1025), (D * D, -65), (training, 128), (heldout, 0)] {
            let mut changed = values.clone();
            changed[index] = invalid;
            assert!(!satisfied(&changed));
        }
    }
}
