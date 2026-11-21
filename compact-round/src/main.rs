mod circuit;
mod private;
mod statement;

use ark_bn254::{Bn254, Fr};
use ark_groth16::{prepare_verifying_key, Groth16, Proof, ProvingKey, VerifyingKey};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem, SynthesisMode};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use circuit::RoundCircuit;
use rand::rngs::OsRng;
use serde_json::json;
use statement::{read, sha256, Result, Statement};
use std::fs::{self, OpenOptions};
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const KEY_LIMIT: u64 = 128 << 20;

fn decode<T: CanonicalDeserialize>(bytes: &[u8]) -> Result<T> {
    let mut raw = bytes;
    let value = T::deserialize_compressed(&mut raw)?;
    if !raw.is_empty() {
        return Err("extra bytes after canonical encoding".into());
    }
    Ok(value)
}

fn write(path: impl AsRef<Path>, value: &impl CanonicalSerialize) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut output = BufWriter::new(file);
    value.serialize_compressed(&mut output)?;
    output.flush()?;
    Ok(())
}

fn setup(directory: &str, circuit: impl ConstraintSynthesizer<Fr>) -> Result<serde_json::Value> {
    let dir = Path::new(directory);
    fs::create_dir(dir)?;
    let key = Groth16::<Bn254>::generate_random_parameters_with_reduction(circuit, &mut OsRng)?;
    let pk = dir.join("proving.key");
    let vk = dir.join("verifying.key");
    write(&pk, &key)?;
    write(&vk, &key.vk)?;
    let bytes = read(vk, 1 << 20)?;
    Ok(
        json!({"proving_key_bytes": fs::metadata(pk)?.len(), "verifying_key_bytes": bytes.len(),
              "verifying_key_sha256": sha256(&bytes)}),
    )
}

fn run(args: &[String]) -> Result<()> {
    let started = Instant::now();
    let report = match args {
        [command] if command == "shape" => {
            let cs = ConstraintSystem::new_ref();
            cs.set_mode(SynthesisMode::Setup);
            RoundCircuit(None).generate_constraints(cs.clone())?;
            json!({"constraints": cs.num_constraints(), "public_inputs": cs.num_instance_variables() - 1,
                   "witnesses": cs.num_witness_variables()})
        }
        [command, directory] if command == "setup" => setup(directory, RoundCircuit(None))?,
        [command, manifest, pin, adapter] if command == "replay" => {
            Statement::load(manifest, pin, adapter)?.replay()?;
            json!({"accepted": true})
        }
        [command, manifest, pin, adapter, proving_key, output] if command == "prove" => {
            let statement = Statement::load(manifest, pin, adapter)?;
            // Reject an incorrect public adapter before generating an artifact.
            statement.replay()?;
            let public = statement.public_inputs();
            let load = Instant::now();
            let key: ProvingKey<Bn254> = decode(&read(proving_key, KEY_LIMIT)?)?;
            let key_seconds = load.elapsed().as_secs_f64();
            let proof_start = Instant::now();
            let proof = Groth16::<Bn254>::create_random_proof_with_reduction(RoundCircuit(Some(public)), &key, &mut OsRng)?;
            let proof_seconds = proof_start.elapsed().as_secs_f64();
            write(output, &proof)?;
            json!({"proof_bytes": fs::metadata(output)?.len(), "key_load_seconds": key_seconds,
                   "prove_seconds": proof_seconds})
        }
        [command, manifest, pin, adapter, verifying_key, key_pin, proof] if command == "verify" => {
            let statement = Statement::load(manifest, pin, adapter)?;
            let public = statement.public_inputs();
            let load = Instant::now();
            let key = read(verifying_key, 1 << 20)?;
            if sha256(&key) != *key_pin {
                return Err("verifying key SHA-256 mismatch".into());
            }
            let vk: VerifyingKey<Bn254> = decode(&key)?;
            let proof: Proof<Bn254> = decode(&read(proof, 1024)?)?;
            let prepared = prepare_verifying_key(&vk);
            let key_prepare_seconds = load.elapsed().as_secs_f64();
            let check = Instant::now();
            if !Groth16::<Bn254>::verify_proof(&prepared, &proof, &public)? {
                return Err("Groth16 rejected the public round and adapter".into());
            }
            json!({"accepted": true, "public_inputs": public.len(), "key_and_proof_prepare_seconds": key_prepare_seconds,
                   "pairing_and_public_inputs_seconds": check.elapsed().as_secs_f64()})
        }
        [command, rest @ ..] if command.starts_with("private-") => private::run(command, rest)?,
        _ => return Err("use shape | setup DIR | replay MANIFEST PIN ADAPTER | prove MANIFEST PIN ADAPTER PK PROOF | verify MANIFEST PIN ADAPTER VK VK_PIN PROOF | private-help".into()),
    };
    println!(
        "RESULT {}",
        json!({"wall_seconds": started.elapsed().as_secs_f64(), "metrics": report})
    );
    Ok(())
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("NO: {error}");
        std::process::exit(1);
    }
}
