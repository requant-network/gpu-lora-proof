mod circuit;
#[cfg(all(feature = "cuda-h", target_os = "linux"))]
mod cuda_h;
mod msm_fixture;
mod prefix;
mod private;
mod private_prefix;
mod statement;

use ark_bn254::{Bn254, Fr};
use ark_groth16::{prepare_verifying_key, Groth16, PreparedVerifyingKey, Proof, ProvingKey, VerifyingKey};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem, SynthesisMode};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use circuit::RoundCircuit;
use prefix::{PrefixCircuit, Shape};
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
    decode_form(bytes, Compress::Yes)
}

fn decode_form<T: CanonicalDeserialize>(bytes: &[u8], compress: Compress) -> Result<T> {
    let mut raw = bytes;
    let value = T::deserialize_with_mode(&mut raw, compress, Validate::Yes)?;
    if !raw.is_empty() {
        return Err("extra bytes after canonical encoding".into());
    }
    Ok(value)
}

fn write(path: impl AsRef<Path>, value: &impl CanonicalSerialize) -> Result<()> {
    write_form(path, value, Compress::Yes)
}

fn write_form(path: impl AsRef<Path>, value: &impl CanonicalSerialize, compress: Compress) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut output = BufWriter::new(file);
    value.serialize_with_mode(&mut output, compress)?;
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

fn load_proving_key(path: &str, compress: Compress) -> Result<(ProvingKey<Bn254>, f64)> {
    let start = Instant::now();
    let key = decode_form(&read(path, KEY_LIMIT)?, compress)?;
    Ok((key, start.elapsed().as_secs_f64()))
}

fn create_proof(
    circuit: impl ConstraintSynthesizer<Fr>,
    key: &ProvingKey<Bn254>,
) -> Result<(Proof<Bn254>, f64)> {
    let start = Instant::now();
    let proof = Groth16::<Bn254>::create_random_proof_with_reduction(circuit, key, &mut OsRng)?;
    Ok((proof, start.elapsed().as_secs_f64()))
}

fn prove(
    circuit: impl ConstraintSynthesizer<Fr>,
    key_path: &str,
    output: &str,
    compress: Compress,
) -> Result<serde_json::Value> {
    let (key, key_seconds) = load_proving_key(key_path, compress)?;
    let (proof, proof_seconds) = create_proof(circuit, &key)?;
    write(output, &proof)?;
    Ok(
        json!({"proof_bytes": fs::metadata(output)?.len(), "key_load_seconds": key_seconds,
              "prove_seconds": proof_seconds}),
    )
}

fn prepare_key(key_path: &str, key_pin: &str) -> Result<PreparedVerifyingKey<Bn254>> {
    let key = read(key_path, 1 << 20)?;
    if sha256(&key) != key_pin {
        return Err("verifying key SHA-256 mismatch".into());
    }
    let vk: VerifyingKey<Bn254> = decode(&key)?;
    Ok(prepare_verifying_key(&vk))
}

fn check_proof(
    public: &[Fr],
    prepared: &PreparedVerifyingKey<Bn254>,
    proof: &Proof<Bn254>,
) -> Result<()> {
    if !Groth16::<Bn254>::verify_proof(prepared, proof, public)? {
        return Err("Groth16 rejected the public round and adapter".into());
    }
    Ok(())
}

fn verify(
    public: &[Fr],
    key_path: &str,
    key_pin: &str,
    proof_path: &str,
) -> Result<serde_json::Value> {
    let load = Instant::now();
    let prepared = prepare_key(key_path, key_pin)?;
    let proof: Proof<Bn254> = decode(&read(proof_path, 1024)?)?;
    let key_prepare_seconds = load.elapsed().as_secs_f64();
    let check = Instant::now();
    check_proof(public, &prepared, &proof)?;
    Ok(
        json!({"accepted": true, "public_inputs": public.len(), "key_and_proof_prepare_seconds": key_prepare_seconds,
              "pairing_and_public_inputs_seconds": check.elapsed().as_secs_f64()}),
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
            statement.replay()?;
            prove(RoundCircuit(Some(statement.public_inputs())), proving_key, output, Compress::Yes)?
        }
        [command, manifest, pin, adapter, verifying_key, key_pin, proof] if command == "verify" => {
            let statement = Statement::load(manifest, pin, adapter)?;
            verify(&statement.public_inputs(), verifying_key, key_pin, proof)?
        }
        [command, shape] if command == "prefix-shape" => {
            let cs = ConstraintSystem::new_ref();
            cs.set_mode(SynthesisMode::Setup);
            PrefixCircuit { shape: Shape::load(shape)?, public: None }.generate_constraints(cs.clone())?;
            json!({"constraints": cs.num_constraints(), "public_inputs": cs.num_instance_variables() - 1,
                   "witnesses": cs.num_witness_variables()})
        }
        [command, shape, directory] if command == "prefix-setup" => {
            setup(directory, PrefixCircuit { shape: Shape::load(shape)?, public: None })?
        }
        [command, shape, manifest, pin, adapter] if command == "prefix-check" => {
            let statement = Statement::load_prefix(manifest, pin, adapter)?;
            check_prefix(&statement, Shape::load(shape)?)?
        }
        [command, shape, manifest, pin, adapter, proving_key, vk_pin, output]
            if command == "prefix-export-h-msm" => {
            let statement = Statement::load_prefix(manifest, pin, adapter)?;
            let shape = Shape::load(shape)?;
            check_prefix(&statement, shape.clone())?;
            let circuit = PrefixCircuit { shape, public: Some(statement.prefix_inputs()) };
            msm_fixture::export_h(circuit, proving_key, vk_pin, output)?
        }
        #[cfg(all(feature = "cuda-h", target_os = "linux"))]
        [command, shape, manifest, pin, adapter, proving_key, vk_pin, output]
            if command == "prefix-prove-cuda-h" || command == "prefix-bench-cuda-h" => {
            let statement = Statement::load_prefix(manifest, pin, adapter)?;
            let shape = Shape::load(shape)?;
            check_prefix(&statement, shape.clone())?;
            let circuit = PrefixCircuit { shape, public: Some(statement.prefix_inputs()) };
            cuda_h::run(circuit, proving_key, vk_pin, output, command == "prefix-bench-cuda-h")?
        }
        [command, source, output] if command == "proving-key-uncompress" => {
            let (key, key_load_seconds) = load_proving_key(source, Compress::Yes)?;
            let output_bytes = key.uncompressed_size();
            if output_bytes as u64 > KEY_LIMIT {
                return Err("uncompressed key exceeds the proving-key byte limit".into());
            }
            write_form(output, &key, Compress::No)?;
            json!({"key_load_seconds": key_load_seconds, "proving_key_bytes": output_bytes})
        }
        [command, shape, manifest, pin, adapter, proving_key, output]
            if command == "prefix-prove" || command == "prefix-prove-uncompressed" => {
            let statement = Statement::load_prefix(manifest, pin, adapter)?;
            let shape = Shape::load(shape)?;
            check_prefix(&statement, shape.clone())?;
            let circuit = PrefixCircuit { shape, public: Some(statement.prefix_inputs()) };
            let compress = if command == "prefix-prove" { Compress::Yes } else { Compress::No };
            prove(circuit, proving_key, output, compress)?
        }
        [command, manifest, pin, adapter, verifying_key, key_pin, proof] if command == "prefix-verify" => {
            let statement = Statement::load_prefix(manifest, pin, adapter)?;
            verify(&statement.prefix_inputs(), verifying_key, key_pin, proof)?
        }
        [command, rest @ ..] if command.starts_with("private-prefix-") => private_prefix::run(command, rest)?,
        [command, rest @ ..] if command.starts_with("private-") => private::run(command, rest)?,
        _ => return Err("use shape | setup DIR | replay MANIFEST PIN ADAPTER | prove MANIFEST PIN ADAPTER PK PROOF | verify MANIFEST PIN ADAPTER VK VK_PIN PROOF | prefix-shape / prefix-setup / prefix-check / prefix-prove / prefix-prove-uncompressed / prefix-verify / prefix-export-h-msm | proving-key-uncompress SOURCE OUTPUT | private-help".into()),
    };
    println!(
        "RESULT {}",
        json!({"wall_seconds": started.elapsed().as_secs_f64(), "metrics": report})
    );
    Ok(())
}

fn check_prefix(statement: &Statement, shape: Shape) -> Result<serde_json::Value> {
    let cs = ConstraintSystem::new_ref();
    PrefixCircuit {
        shape,
        public: Some(statement.prefix_inputs()),
    }
    .generate_constraints(cs.clone())?;
    if !cs.is_satisfied()? {
        return Err("prefix circuit rejected the statement".into());
    }
    Ok(json!({"accepted": true, "constraints": cs.num_constraints()}))
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("NO: {error}");
        std::process::exit(1);
    }
}
