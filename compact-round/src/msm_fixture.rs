use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField, Zero};
use ark_groth16::r1cs_to_qap::{LibsnarkReduction, R1CSToQAP};
use ark_poly::GeneralEvaluationDomain;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem, OptimizationGoal};
use ark_serialize::{CanonicalSerialize, Compress};
use serde_json::json;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use crate::{load_proving_key, sha256, Result};

pub fn export_h(
    circuit: impl ConstraintSynthesizer<Fr>,
    proving_key: &str,
    vk_pin: &str,
    path: &str,
) -> Result<serde_json::Value> {
    let (key, _) = load_proving_key(proving_key, Compress::Yes)?;
    let mut vk = Vec::new();
    key.vk.serialize_compressed(&mut vk)?;
    if sha256(&vk) != vk_pin {
        return Err("proving key has a different verification-key pin".into());
    }

    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    circuit.generate_constraints(cs.clone())?;
    cs.finalize();
    let h = LibsnarkReduction::witness_map::<Fr, GeneralEvaluationDomain<Fr>>(cs)?;
    let pairs = key.h_query.len();
    if pairs == 0 || h.len() != pairs + 1 || !h[pairs].is_zero() {
        return Err("unexpected H query length or trailing coefficient".into());
    }

    let mut out = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    for point in &key.h_query {
        if point.infinity {
            out.write_all(&[0; 64])?;
        } else {
            out.write_all(&point.x.into_bigint().to_bytes_le())?;
            out.write_all(&point.y.into_bigint().to_bytes_le())?;
        }
    }
    for scalar in &h[..pairs] {
        out.write_all(&scalar.into_bigint().to_bytes_le())?;
    }
    out.flush()?;
    Ok(json!({"pairs": pairs, "witness_coefficients": h.len(), "bytes": pairs * 96}))
}
