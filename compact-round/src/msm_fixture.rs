use ark_bn254::{Bn254, Fr, G1Affine};
use ark_ff::{BigInt, PrimeField, Zero};
use ark_groth16::r1cs_to_qap::{LibsnarkReduction, R1CSToQAP};
use ark_groth16::ProvingKey;
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem, OptimizationGoal};
use ark_serialize::{CanonicalSerialize, Compress};
use serde_json::json;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use crate::{load_proving_key, sha256, Result};

pub fn check_key_pin(key: &ProvingKey<Bn254>, vk_pin: &str) -> Result<()> {
    let mut vk = Vec::new();
    key.vk.serialize_compressed(&mut vk)?;
    if sha256(&vk) != vk_pin {
        return Err("proving key has a different verification-key pin".into());
    }
    Ok(())
}

pub fn h_scalars(h: &[Fr], pairs: usize) -> Result<&[Fr]> {
    if pairs == 0 || h.len() != pairs + 1 || !h[pairs].is_zero() {
        return Err("unexpected H query length or trailing coefficient".into());
    }
    Ok(&h[..pairs])
}

pub fn field_bytes(integer: BigInt<4>) -> [u8; 32] {
    let mut wire = [0; 32];
    for (bytes, limb) in wire.chunks_exact_mut(8).zip(integer.0) {
        bytes.copy_from_slice(&limb.to_le_bytes());
    }
    wire
}

pub fn point_bytes(point: &G1Affine) -> [u8; 64] {
    let mut wire = [0; 64];
    if !point.infinity {
        wire[..32].copy_from_slice(&field_bytes(point.x.into_bigint()));
        wire[32..].copy_from_slice(&field_bytes(point.y.into_bigint()));
    }
    wire
}

pub fn export_h(
    circuit: impl ConstraintSynthesizer<Fr>,
    proving_key: &str,
    vk_pin: &str,
    path: &str,
) -> Result<serde_json::Value> {
    let (key, _) = load_proving_key(proving_key, Compress::Yes)?;
    check_key_pin(&key, vk_pin)?;

    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    circuit.generate_constraints(cs.clone())?;
    cs.finalize();
    let h = LibsnarkReduction::witness_map::<Fr, GeneralEvaluationDomain<Fr>>(cs)?;
    let pairs = key.h_query.len();
    let scalars = h_scalars(&h, pairs)?;

    let mut out = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    for point in &key.h_query {
        out.write_all(&point_bytes(point))?;
    }
    for scalar in scalars {
        out.write_all(&field_bytes(scalar.into_bigint()))?;
    }
    out.flush()?;
    Ok(json!({"pairs": pairs, "witness_coefficients": h.len(), "bytes": pairs * 96}))
}
