use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::R1CSVar;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use serde::Deserialize;
use std::collections::HashSet;

use crate::circuit::{constant, forward, magnitude, residual, sum, update};
use crate::statement::{read, scalar, ADAPTER, BATCH, D, HELDOUT, PUBLIC, R, SCALE};

type Var = FpVar<Fr>;
type Bit = Boolean<Fr>;
type Synthesis<T> = std::result::Result<T, SynthesisError>;

const LOSS_BITS: usize = 68;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub orders: [Vec<usize>; 2],
    pub labels: [usize; HELDOUT],
}

impl Shape {
    pub fn load(path: &str) -> crate::statement::Result<Self> {
        let shape: Self = serde_json::from_slice(&read(path, 8 << 10)?)?;
        for order in &shape.orders {
            if order.iter().any(|&index| index >= ADAPTER)
                || order.iter().collect::<HashSet<_>>().len() != order.len()
            {
                return Err("invalid active coordinate order".into());
            }
        }
        if shape.labels.iter().any(|&label| label >= D) {
            return Err("invalid held-out label index".into());
        }
        Ok(shape)
    }
}

pub struct PrefixCircuit {
    pub shape: Shape,
    pub public: Option<Vec<Fr>>,
}

// The full range is far below the BN254 scalar modulus. Negative field residues
// therefore cannot satisfy the reconstruction with this number of bits.
fn nonnegative(value: &Var, width: usize) -> Synthesis<()> {
    let assigned = value.value().ok().map(|field| field.into_bigint());
    let bits = (0..width)
        .map(|i| {
            Bit::new_witness(value.cs(), || {
                let value = assigned.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
                Ok(value.as_ref()[i / 64] & (1u64 << (i % 64)) != 0)
            })
        })
        .collect::<Synthesis<Vec<_>>>()?;
    Bit::le_bits_to_fp(&bits)?.enforce_equal(value)
}

struct TrainingRow {
    features: Vec<Var>,
    latent: Vec<Var>,
    errors: Vec<Var>,
}

fn prefix_transition(
    phase: usize,
    a: &[Var],
    b: &[Var],
    weights: &[Var],
    inputs: Vec<(Vec<Var>, Vec<Var>)>,
    order: &[usize],
) -> Synthesis<Vec<Var>> {
    let mut total = vec![constant(0); ADAPTER];
    let mut rows = Vec::new();
    for (x, target) in inputs {
        let (u, y) = forward(&x, weights, a, b);
        let g = y
            .iter()
            .zip(&target)
            .map(|(y, t)| residual(y, t))
            .collect::<Synthesis<Vec<_>>>()?;
        if phase == 0 {
            for j in 0..D {
                for k in 0..R {
                    total[j * R + k] += (&g[j] * &u[k]) * scalar(SCALE);
                }
            }
        } else {
            for k in 0..R {
                let q = sum((0..D).map(|j| &g[j] * &b[j * R + k]));
                for j in 0..D {
                    total[k * D + j] += (&q * &x[j]) * scalar(SCALE);
                }
            }
        }
        rows.push(TrainingRow {
            features: x,
            latent: u,
            errors: y.iter().zip(&target).map(|(y, t)| y - t).collect(),
        });
    }

    let old = if phase == 0 { b } else { a };
    let deltas = old
        .iter()
        .zip(&total)
        .map(|(old, gradient)| Ok(update(old, gradient)? - old))
        .collect::<Synthesis<Vec<_>>>()?;
    let active: HashSet<_> = order.iter().copied().collect();
    for (index, delta) in deltas.iter().enumerate() {
        if active.contains(&index) {
            delta.enforce_not_equal(&constant(0))?;
        } else {
            delta.enforce_equal(&constant(0))?;
        }
    }

    let mut last = None;
    let mut losses = vec![loss(&rows)];
    for &index in order {
        let (_, _, magnitude) = magnitude(&total[index], 32)?;
        let priority = magnitude * scalar(ADAPTER as i64) + scalar((ADAPTER - 1 - index) as i64);
        if let Some(previous) = last {
            nonnegative(&(previous - &priority - constant(1)), 40)?;
        }
        last = Some(priority);
        let delta = &deltas[index] * scalar(SCALE);
        for row in &mut rows {
            if phase == 0 {
                row.errors[index / R] += &row.latent[index % R] * &delta;
            } else {
                let change = &row.features[index % D] * &delta;
                for (j, error) in row.errors.iter_mut().enumerate() {
                    *error += &b[j * R + index / D] * &change;
                }
            }
        }
        losses.push(loss(&rows));
    }

    let selected = first_minimum(&losses)?;
    let mut result = old.to_vec();
    let mut included = constant(0);
    for (position, &index) in order.iter().enumerate().rev() {
        included += &selected[position + 1];
        result[index] += &deltas[index] * &included;
    }
    Ok(result)
}

fn loss(rows: &[TrainingRow]) -> Var {
    sum(rows
        .iter()
        .flat_map(|row| row.errors.iter().map(|error| error * error)))
}

fn first_minimum(losses: &[Var]) -> Synthesis<Vec<Var>> {
    let chosen = losses
        .iter()
        .map(|loss| Ok(loss.value()?.into_bigint()))
        .collect::<Synthesis<Vec<_>>>()
        .ok()
        .and_then(|values| {
            values
                .into_iter()
                .enumerate()
                .min_by_key(|&(i, value)| (value, i))
        })
        .map(|(i, _)| i);
    let cs = losses[0].cs();
    let select = (0..losses.len())
        .map(|j| {
            Ok(Var::from(Bit::new_witness(cs.clone(), || {
                Ok(j == chosen.ok_or(SynthesisError::AssignmentMissing)?)
            })?))
        })
        .collect::<Synthesis<Vec<_>>>()?;
    sum(select.iter().cloned()).enforce_equal(&constant(1))?;

    let best = sum(losses.iter().zip(&select).map(|(value, pick)| value * pick));
    let mut later = constant(1);
    for (value, pick) in losses.iter().zip(&select) {
        later -= pick;
        nonnegative(&(value - &best - &later), LOSS_BITS)?;
    }
    Ok(select)
}

impl ConstraintSynthesizer<Fr> for PrefixCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Synthesis<()> {
        let public = (0..PUBLIC + 1)
            .map(|i| {
                Var::new_input(cs.clone(), || {
                    self.public
                        .as_ref()
                        .map(|values| values[i])
                        .ok_or(SynthesisError::AssignmentMissing)
                })
            })
            .collect::<Synthesis<Vec<_>>>()?;
        for pin in &public[..2] {
            pin.enforce_equal(&Var::new_witness(cs.clone(), || pin.value())?)?;
        }
        let mut remaining = public[2..].iter();
        let mut take = |count| remaining.by_ref().take(count).cloned().collect::<Vec<_>>();
        let w = take(D * D);
        let mut a = take(ADAPTER);
        let mut b = take(ADAPTER);
        for (phase, order) in self.shape.orders.iter().enumerate() {
            let mut rows = Vec::new();
            for _ in 0..2 {
                let x = take(BATCH * D);
                let targets = take(BATCH * D);
                rows.extend(
                    x.chunks_exact(D)
                        .zip(targets.chunks_exact(D))
                        .map(|(x, t)| (x.to_vec(), t.to_vec())),
                );
            }
            let result = prefix_transition(phase, &a, &b, &w, rows, order)?;
            if phase == 0 {
                b = result;
            } else {
                a = result;
            }
        }

        let heldout = take(HELDOUT * D);
        let targets = take(HELDOUT * D);
        let mut squared_error = constant(0);
        for ((x, target), &label) in heldout
            .chunks_exact(D)
            .zip(targets.chunks_exact(D))
            .zip(&self.shape.labels)
        {
            let (_, predicted) = forward(x, &w, &a, &b);
            for j in 0..D {
                if j != label {
                    nonnegative(&(&target[label] - &target[j] - constant(1)), 9)?;
                    nonnegative(
                        &(&predicted[label] - &predicted[j] - constant(i64::from(j < label))),
                        32,
                    )?;
                }
                let error = &predicted[j] - &target[j];
                squared_error += &error * &error;
            }
        }
        for (value, expected) in a.iter().chain(&b).zip(take(2 * ADAPTER)) {
            value.enforce_equal(&expected)?;
        }
        nonnegative(&(&take(1)[0] - squared_error), LOSS_BITS)?;
        debug_assert!(remaining.next().is_none());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn picks_first_prefix_with_minimum_loss() {
        let cs = ConstraintSystem::new_ref();
        let losses: Vec<_> = [9, 8, 7, 7, 8]
            .into_iter()
            .map(|loss| Var::new_input(cs.clone(), || Ok(scalar(loss))).unwrap())
            .collect();
        let selected = first_minimum(&losses).unwrap();
        for (i, pick) in selected.iter().enumerate() {
            assert_eq!(pick.value().unwrap(), scalar(i64::from(i == 2)));
        }
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn unsigned_gate_rejects_a_negative_difference() {
        let cs = ConstraintSystem::new_ref();
        let difference = Var::new_input(cs.clone(), || Ok(scalar(-1))).unwrap();
        nonnegative(&difference, LOSS_BITS).unwrap();
        assert!(!cs.is_satisfied().unwrap());
    }
}
