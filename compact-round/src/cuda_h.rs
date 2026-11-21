use ark_bn254::{Bn254, Fq, Fr, G1Affine};
use ark_ec::CurveGroup;
use ark_ff::{BigInt, PrimeField, UniformRand};
use ark_groth16::r1cs_to_qap::{LibsnarkReduction, R1CSToQAP};
use ark_groth16::{prepare_verifying_key, Groth16, Proof, ProvingKey};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{
    ConstraintMatrices, ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef,
    OptimizationGoal, Result as Synthesis, SynthesisError,
};
use ark_serialize::{CanonicalSerialize, Compress};
use rand::rngs::OsRng;
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};

use crate::msm_fixture::{check_key_pin, field_bytes, h_scalars, point_bytes};
use crate::{check_proof, load_proving_key, write, PrefixCircuit, Result};

#[link(name = "fullmsm")]
extern "C" {
    fn full_msm(
        bases: *const u8,
        scalars: *const u8,
        count: i32,
        width: i32,
        repeats: i32,
        point: *mut u8,
        timings: *mut f32,
    ) -> i32;
}

fn coord(bytes: &[u8]) -> Result<Fq> {
    let mut limbs = [0; 4];
    for (limb, raw) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
        *limb = u64::from_le_bytes(raw.try_into()?);
    }
    Fq::from_bigint(BigInt::new(limbs)).ok_or_else(|| "noncanonical CUDA H coordinate".into())
}

fn h_point(bytes: &[u8; 64]) -> Result<G1Affine> {
    if bytes.iter().all(|b| *b == 0) {
        return Ok(G1Affine::identity());
    }
    let point = G1Affine::new_unchecked(coord(&bytes[..32])?, coord(&bytes[32..])?);
    if !point.is_on_curve() || !point.is_in_correct_subgroup_assuming_on_curve() {
        return Err("CUDA returned an invalid H point".into());
    }
    Ok(point)
}

// This adapter enters the upstream proof assembler after H has been evaluated.
// It leaves the H contribution at the group identity until C is completed below.
struct AssemblerWithoutH;

impl R1CSToQAP for AssemblerWithoutH {
    fn instance_map_with_evaluation<F: PrimeField, D: EvaluationDomain<F>>(
        _: ConstraintSystemRef<F>,
        _: &F,
    ) -> Synthesis<(Vec<F>, Vec<F>, Vec<F>, F, usize, usize)> {
        Err(SynthesisError::Unsatisfiable)
    }

    fn witness_map_from_matrices<F: PrimeField, D: EvaluationDomain<F>>(
        _: &ConstraintMatrices<F>,
        _: usize,
        _: usize,
        _: &[F],
    ) -> Synthesis<Vec<F>> {
        Ok(Vec::new())
    }

    fn h_query_scalars<F: PrimeField, D: EvaluationDomain<F>>(
        _: usize,
        _: F,
        _: F,
        _: F,
    ) -> Synthesis<Vec<F>> {
        Err(SynthesisError::Unsatisfiable)
    }
}

fn prove(
    circuit: PrefixCircuit,
    key: &ProvingKey<Bn254>,
    bases: &[u8],
    r: Fr,
    s: Fr,
) -> Result<(Proof<Bn254>, Value)> {
    let started = Instant::now();
    let cs = ConstraintSystem::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);
    circuit.generate_constraints(cs.clone())?;
    cs.finalize();
    let inputs = cs.num_instance_variables();
    let constraints = cs.num_constraints();
    let matrices = cs.to_matrices().ok_or("missing constraint matrices")?;
    let assignment = {
        let state = cs.borrow().ok_or("missing constraint assignment")?;
        [
            &state.instance_assignment[..],
            &state.witness_assignment[..],
        ]
        .concat()
    };
    let synth_elapsed = started.elapsed().as_secs_f64();
    let h = LibsnarkReduction::witness_map_from_matrices::<Fr, GeneralEvaluationDomain<Fr>>(
        &matrices,
        inputs,
        constraints,
        &assignment,
    )?;
    let witness_elapsed = started.elapsed().as_secs_f64() - synth_elapsed;
    let coefficients: Vec<u8> = h_scalars(&h, key.h_query.len())?
        .iter()
        .flat_map(|scalar| field_bytes(scalar.into_bigint()))
        .collect();
    let count = i32::try_from(coefficients.len() / 32)?;
    if bases.len() != count as usize * 64 {
        return Err("mismatched CUDA H input buffers".into());
    }
    let mut wire = [0u8; 64];
    let mut upload_ms = 0f32;
    let cuda_start = Instant::now();
    // The checked lengths match the ABI. Zero repeats writes one timing and
    // one point, and the native call synchronizes before returning.
    let code = unsafe {
        full_msm(
            bases.as_ptr(),
            coefficients.as_ptr(),
            count,
            8,
            0,
            wire.as_mut_ptr(),
            &mut upload_ms,
        )
    };
    let cuda_seconds = cuda_start.elapsed().as_secs_f64();
    if code != 0 {
        return Err(format!("CUDA H MSM failed with status {code}").into());
    }
    let h = h_point(&wire)?;
    let assemble_start = Instant::now();
    let mut proof = Groth16::<Bn254, AssemblerWithoutH>::create_proof_with_reduction_and_matrices(
        key,
        r,
        s,
        &matrices,
        inputs,
        constraints,
        &assignment,
    )?;
    proof.c = (proof.c + h).into_affine();
    Ok((
        proof,
        json!({
            "prove_seconds": started.elapsed().as_secs_f64(), "synthesis_seconds": synth_elapsed,
            "witness_seconds": witness_elapsed, "h_cuda_wall_seconds": cuda_seconds,
            "h_upload_and_import_ms": upload_ms, "proof_assembly_seconds": assemble_start.elapsed().as_secs_f64(),
        }),
    ))
}

pub fn run(
    circuit: PrefixCircuit,
    proving_key: &str,
    vk_pin: &str,
    output: &str,
    benchmark: bool,
) -> Result<Value> {
    let (key, load_seconds) = load_proving_key(proving_key, Compress::No)?;
    check_key_pin(&key, vk_pin)?;
    let pvk = prepare_verifying_key(&key.vk);
    let public = circuit
        .public
        .as_deref()
        .ok_or("missing public prefix inputs")?;
    let enc_start = Instant::now();
    let bases: Vec<u8> = key.h_query.iter().flat_map(point_bytes).collect();
    let encode_seconds = enc_start.elapsed().as_secs_f64();
    let new_circuit = || PrefixCircuit {
        shape: circuit.shape.clone(),
        public: circuit.public.clone(),
    };
    let path = Path::new(output);
    let mut rng = OsRng;
    let mut rows = Vec::new();
    let mut proofs = Vec::new();
    for round in 0..if benchmark { 7 } else { 1 } {
        let r = Fr::rand(&mut rng);
        let s = Fr::rand(&mut rng);
        let cpu = || -> Result<_> {
            let start = Instant::now();
            let proof = Groth16::<Bn254>::create_proof_with_reduction(new_circuit(), &key, r, s)?;
            Ok((proof, start.elapsed().as_secs_f64()))
        };
        let gpu = || prove(new_circuit(), &key, &bases, r, s);
        let (cpu_result, (gpu_proof, metrics)) = if !benchmark {
            (None, gpu()?)
        } else if round % 2 == 0 {
            (Some(cpu()?), gpu()?)
        } else {
            let first = gpu()?;
            (Some(cpu()?), first)
        };
        let verify_start = Instant::now();
        check_proof(public, &pvk, &gpu_proof)?;
        let verify_seconds = verify_start.elapsed().as_secs_f64();
        let cpu_seconds = if let Some((cpu_proof, seconds)) = cpu_result {
            let mut stock = Vec::new();
            let mut cuda = Vec::new();
            cpu_proof.serialize_compressed(&mut stock)?;
            gpu_proof.serialize_compressed(&mut cuda)?;
            if stock != cuda {
                return Err("CUDA proof bytes differ from the stock prover".into());
            }
            Some(seconds)
        } else {
            None
        };
        rows.push(json!({"round": round, "cpu_prove_seconds": cpu_seconds,
                         "cuda_verify_seconds": verify_seconds, "cuda": metrics}));
        proofs.push(gpu_proof);
    }
    if benchmark {
        fs::create_dir(path)?;
        for (i, proof) in proofs.iter().enumerate() {
            write(path.join(format!("{i}.proof")), proof)?;
        }
    } else {
        write(path, &proofs[0])?;
    }
    Ok(
        json!({"key_load_seconds": load_seconds, "encode_h_bases_seconds": encode_seconds,
              "pairs": key.h_query.len(), "trials": rows}),
    )
}
