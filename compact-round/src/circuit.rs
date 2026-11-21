use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::{fp::FpVar, FieldVar};
use ark_r1cs_std::select::CondSelectGadget;
use ark_r1cs_std::R1CSVar;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};

use crate::statement::{scalar, ADAPTER, BATCH, D, HELDOUT, PUBLIC, R, SCALE};

type Var = FpVar<Fr>;
type Bit = Boolean<Fr>;
type Result<T> = std::result::Result<T, SynthesisError>;

pub struct RoundCircuit(pub Option<Vec<Fr>>);

fn constant(v: i64) -> Var {
    Var::Constant(scalar(v))
}

fn sum(values: impl Iterator<Item = Var>) -> Var {
    values.fold(constant(0), |total, value| total + value)
}

// Both integer signs are represented by field elements far below the modulus.
fn magnitude(value: &Var, width: usize) -> Result<(Bit, Vec<Bit>, Var)> {
    let assigned = value.value().ok().and_then(|v| {
        [(false, v), (true, -v)]
            .into_iter()
            .find_map(|(negative, v)| {
                let repr = v.into_bigint();
                let limbs: &[u64] = repr.as_ref();
                (limbs[0] < (1u64 << width) && limbs[1..].iter().all(|&x| x == 0))
                    .then_some((negative, limbs[0]))
            })
    });
    let cs = value.cs();
    let given = || assigned.ok_or(SynthesisError::AssignmentMissing);
    let sign = Bit::new_witness(cs.clone(), || Ok(given()?.0))?;
    let bits: Vec<_> = (0..width)
        .map(|i| Bit::new_witness(cs.clone(), || Ok(given()?.1 & (1 << i) != 0)))
        .collect::<Result<_>>()?;
    let mag = Bit::le_bits_to_fp(&bits)?;
    signed(&sign, &mag)?.enforce_equal(value)?;
    Ok((sign, bits, mag))
}

fn signed(sign: &Bit, mag: &Var) -> Result<Var> {
    Var::conditionally_select(sign, &mag.negate()?, mag)
}

fn residual(predicted: &Var, target: &Var) -> Result<Var> {
    let (sign, bits, mag) = magnitude(&(predicted - target), 32)?;
    let saturated = Bit::kary_or(&bits[7..])?;
    let clipped = Var::conditionally_select(&saturated, &constant(127), &mag)?;
    signed(&sign, &clipped)
}

fn update(old: &Var, gradient: &Var) -> Result<Var> {
    let (negative, bits, _) = magnitude(gradient, 32)?;
    let nonzero = Var::from(Bit::kary_or(&bits)?);
    let new = old - signed(&negative, &nonzero)?;
    let (sign, bits, abs) = magnitude(&new, 8)?;
    let over = Bit::kary_or(&[
        bits[7].clone(),
        Bit::kary_and(&[bits[6].clone(), Bit::kary_or(&bits[..6])?])?,
    ])?;
    signed(
        &sign,
        &Var::conditionally_select(&over, &constant(64), &abs)?,
    )
}

fn forward(x: &[Var], w: &[Var], a: &[Var], b: &[Var]) -> (Vec<Var>, Vec<Var>) {
    let u: Vec<_> = a
        .chunks_exact(D)
        .map(|row| sum(x.iter().zip(row).map(|(x, a)| x * a)))
        .collect();
    let y = (0..D)
        .map(|j| {
            let base = sum((0..D).map(|i| &x[i] * &w[i * D + j]));
            let delta = sum(u.iter().zip(&b[j * R..][..R]).map(|(u, b)| u * b));
            base + delta * scalar(SCALE)
        })
        .collect();
    (u, y)
}

impl ConstraintSynthesizer<Fr> for RoundCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<()> {
        let public = (0..PUBLIC)
            .map(|i| {
                Var::new_input(cs.clone(), || {
                    self.0
                        .as_ref()
                        .map(|values| values[i])
                        .ok_or(SynthesisError::AssignmentMissing)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for pin in &public[..2] {
            // Bind both public halves of the manifest's exact SHA-256.
            pin.enforce_equal(&Var::new_witness(cs.clone(), || pin.value())?)?;
        }
        let mut remaining = public[2..].iter();
        let mut take = |count| remaining.by_ref().take(count).cloned().collect::<Vec<_>>();
        let w = take(D * D);
        let mut a = take(ADAPTER);
        let mut b = take(ADAPTER);
        for phase in 0..2 {
            let mut combined = vec![constant(0); ADAPTER];
            for _ in 0..2 {
                let x = take(BATCH * D);
                let target = take(BATCH * D);
                for (row, target) in x.chunks_exact(D).zip(target.chunks_exact(D)) {
                    let (u, y) = forward(row, &w, &a, &b);
                    let g: Vec<_> = y
                        .iter()
                        .zip(target)
                        .map(|(p, t)| residual(p, t))
                        .collect::<Result<_>>()?;
                    if phase == 0 {
                        for j in 0..D {
                            for k in 0..R {
                                combined[j * R + k] += (&g[j] * &u[k]) * scalar(SCALE);
                            }
                        }
                    } else {
                        for k in 0..R {
                            let q = sum((0..D).map(|j| &g[j] * &b[j * R + k]));
                            for j in 0..D {
                                combined[k * D + j] += (&q * &row[j]) * scalar(SCALE);
                            }
                        }
                    }
                }
            }
            let factor = if phase == 0 { &mut b } else { &mut a };
            for (old, g) in factor.iter_mut().zip(combined) {
                *old = update(old, &g)?;
            }
        }
        let x = take(HELDOUT * D);
        let target = take(HELDOUT * D);
        for (row, target) in x.chunks_exact(D).zip(target.chunks_exact(D)) {
            for (y, target) in forward(row, &w, &a, &b).1.iter().zip(target) {
                y.enforce_equal(target)?;
            }
        }
        for (value, expected) in a.iter().chain(&b).zip(take(2 * ADAPTER)) {
            value.enforce_equal(&expected)?;
        }
        debug_assert!(remaining.next().is_none());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn signed_gradient_clips_on_both_boundaries() {
        let cs = ConstraintSystem::new_ref();
        for x in [-1_000_000, -128, -127, -126, 0, 126, 127, 128, 1_000_000] {
            let v = Var::new_input(cs.clone(), || Ok(scalar(x))).unwrap();
            let g = residual(&v, &constant(0)).unwrap();
            assert_eq!(g.value().unwrap(), scalar(x.clamp(-127, 127)));
        }
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn sign_step_clips_and_preserves_zero_gradient() {
        let cs = ConstraintSystem::new_ref();
        for old in [-64, -63, 0, 63, 64] {
            for g in [-2_000_000_000i64, -1, 0, 1, 2_000_000_000] {
                let old_var = Var::new_input(cs.clone(), || Ok(scalar(old))).unwrap();
                let grad = Var::new_witness(cs.clone(), || Ok(scalar(g))).unwrap();
                assert_eq!(
                    update(&old_var, &grad).unwrap().value().unwrap(),
                    scalar((old - g.signum()).clamp(-64, 64))
                );
            }
        }
        assert!(cs.is_satisfied().unwrap());
    }
}
