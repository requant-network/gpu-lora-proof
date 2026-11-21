use ark_bn254::Fr;
use ark_ff::PrimeField;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const D: usize = 32;
pub const R: usize = 4;
pub const BATCH: usize = 4;
pub const HELDOUT: usize = 16;
pub const SCALE: i64 = 8;
pub const ADAPTER: usize = D * R;
pub const PUBLIC: usize = 2 + D * D + 4 * ADAPTER + 8 * BATCH * D + 2 * HELDOUT * D;
const SCHEMA: &str = "gpu-lora-round/v1";
const PREFIX_SCHEMA: &str = "gpu-lora-round/v2";

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "UPPERCASE")]
pub struct Adapter {
    pub a: Vec<i64>,
    pub b: Vec<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Slot {
    worker: String,
    x: Vec<i64>,
    target: Vec<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Phase {
    factor: String,
    slots: [Slot; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Batch {
    x: Vec<i64>,
    target: Vec<i64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    schema: String,
    id: String,
    #[serde(rename = "W")]
    w: Vec<i64>,
    initial: Adapter,
    phases: [Phase; 2],
    heldout: Batch,
    max_heldout_sse: u64,
    #[serde(default, deserialize_with = "read_floor")]
    min_heldout_correct: Option<u64>,
}

fn read_floor<'de, T: serde::Deserializer<'de>>(
    reader: T,
) -> std::result::Result<Option<u64>, T::Error> {
    Ok(Some(u64::deserialize(reader)?))
}

pub struct Statement {
    manifest: Manifest,
    pin: [u8; 32],
    pub output: Adapter,
}

pub fn read(path: impl AsRef<Path>, limit: u64) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut data)?;
    if data.len() as u64 > limit {
        return Err("file exceeds size limit".into());
    }
    Ok(data)
}

pub fn sha256(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

pub fn scalar(value: i64) -> Fr {
    let absolute = Fr::from(value.unsigned_abs());
    if value < 0 {
        -absolute
    } else {
        absolute
    }
}

fn bounded(v: &[i64], n: usize, max: u64) -> bool {
    v.len() == n && v.iter().all(|v| v.unsigned_abs() <= max)
}

impl Adapter {
    pub(crate) fn load(path: &str) -> Result<Self> {
        let adapter: Self = serde_json::from_slice(&read(path, 8 << 10)?)?;
        if !adapter.valid() {
            return Err("invalid adapter range or shape".into());
        }
        Ok(adapter)
    }

    fn valid(&self) -> bool {
        bounded(&self.a, ADAPTER, 64) && bounded(&self.b, ADAPTER, 64)
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = Fr> + '_ {
        self.a.iter().chain(&self.b).copied().map(scalar)
    }
}

impl Manifest {
    pub(crate) fn load(path: &str, pin: &str) -> Result<(Self, [u8; 32])> {
        Self::load_schema(path, pin, SCHEMA)
    }

    pub(crate) fn load_prefix(path: &str, pin: &str) -> Result<(Self, [u8; 32])> {
        Self::load_schema(path, pin, PREFIX_SCHEMA)
    }

    pub(crate) fn heldout_ceiling(&self) -> Fr {
        Fr::from(self.max_heldout_sse)
    }

    fn load_schema(path: &str, pin: &str, schema: &str) -> Result<(Self, [u8; 32])> {
        let bytes = read(path, 64 << 10)?;
        if sha256(&bytes) != pin {
            return Err("manifest SHA-256 mismatch".into());
        }
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        if manifest.schema != schema
            || manifest.id.is_empty()
            || manifest.id.len() > 128
            || (schema == SCHEMA
                && (manifest.max_heldout_sse != 0 || manifest.min_heldout_correct.is_some()))
            || (schema == PREFIX_SCHEMA && manifest.min_heldout_correct != Some(HELDOUT as u64))
            || !bounded(&manifest.w, D * D, 1024)
            || !manifest.initial.valid()
        {
            return Err("invalid round statement or unsupported held-out gate".into());
        }
        let mut train = HashSet::new();
        for (phase, expected) in manifest.phases.iter().zip(["B", "A"]) {
            if phase.factor != expected {
                return Err("incorrect optimizer schedule".into());
            }
            for (slot, worker) in phase.slots.iter().zip(["worker-a", "worker-b"]) {
                if slot.worker != worker
                    || !bounded(&slot.x, BATCH * D, 127)
                    || !bounded(&slot.target, BATCH * D, 127)
                {
                    return Err("invalid training slot".into());
                }
                train.extend(slot.x.chunks_exact(D));
            }
        }
        let x = &manifest.heldout.x;
        let unique: HashSet<_> = x.chunks_exact(D).collect();
        if !bounded(x, HELDOUT * D, 127)
            || !bounded(&manifest.heldout.target, HELDOUT * D, 127)
            || unique.len() != HELDOUT
            || !unique.is_disjoint(&train)
        {
            return Err("invalid or overlapping held-out rows".into());
        }
        Ok((manifest, Sha256::digest(bytes).into()))
    }

    pub(crate) fn order_fields(&self) -> Vec<Fr> {
        let mut out = Vec::new();
        let mut push = |values: &[i64]| out.extend(values.iter().copied().map(scalar));
        push(&self.w);
        push(&self.initial.a);
        push(&self.initial.b);
        for phase in &self.phases {
            for slot in &phase.slots {
                push(&slot.x);
                push(&slot.target);
            }
        }
        push(&self.heldout.x);
        push(&self.heldout.target);
        out
    }
}

impl Statement {
    pub fn load(manifest_path: &str, pin: &str, output_path: &str) -> Result<Self> {
        let (manifest, pin) = Manifest::load(manifest_path, pin)?;
        let output = Adapter::load(output_path)?;
        Ok(Self {
            manifest,
            pin,
            output,
        })
    }

    pub fn load_prefix(manifest_path: &str, pin: &str, output_path: &str) -> Result<Self> {
        let (manifest, pin) = Manifest::load_prefix(manifest_path, pin)?;
        let output = Adapter::load(output_path)?;
        Ok(Self {
            manifest,
            pin,
            output,
        })
    }

    pub fn prefix_inputs(&self) -> Vec<Fr> {
        let mut fields = self.public_inputs();
        fields.push(Fr::from(self.manifest.max_heldout_sse));
        fields
    }

    pub fn public_inputs(&self) -> Vec<Fr> {
        let mut out: Vec<_> = self
            .pin
            .chunks_exact(16)
            .map(Fr::from_le_bytes_mod_order)
            .collect();
        out.extend(self.manifest.order_fields());
        out.extend(self.output.values());
        debug_assert_eq!(out.len(), PUBLIC);
        out
    }

    pub fn replay(&self) -> Result<()> {
        let m = &self.manifest;
        let mut adapter = m.initial.clone();
        for (i, phase) in m.phases.iter().enumerate() {
            let mut summed = vec![0i64; ADAPTER];
            for slot in &phase.slots {
                for (x, target) in slot.x.chunks_exact(D).zip(slot.target.chunks_exact(D)) {
                    let (u, y) = predict(x, &m.w, &adapter);
                    let g: Vec<_> = y
                        .iter()
                        .zip(target)
                        .map(|(p, t)| (p - t).clamp(-127, 127))
                        .collect();
                    if i == 0 {
                        for j in 0..D {
                            for k in 0..R {
                                summed[j * R + k] += SCALE * g[j] * u[k];
                            }
                        }
                    } else {
                        for k in 0..R {
                            let q = (0..D).map(|j| g[j] * adapter.b[j * R + k]).sum::<i64>();
                            for j in 0..D {
                                summed[k * D + j] += SCALE * q * x[j];
                            }
                        }
                    }
                }
            }
            let factor = if i == 0 {
                &mut adapter.b
            } else {
                &mut adapter.a
            };
            for (weight, g) in factor.iter_mut().zip(summed) {
                *weight = (*weight - g.signum()).clamp(-64, 64);
            }
        }
        if adapter != self.output
            || m.heldout
                .x
                .chunks_exact(D)
                .zip(m.heldout.target.chunks_exact(D))
                .any(|(x, target)| predict(x, &m.w, &adapter).1 != target)
        {
            return Err("adapter or held-out challenge failed direct replay".into());
        }
        Ok(())
    }
}

fn predict(x: &[i64], w: &[i64], adapter: &Adapter) -> (Vec<i64>, Vec<i64>) {
    let u: Vec<_> = adapter
        .a
        .chunks_exact(D)
        .map(|row| x.iter().zip(row).map(|(a, b)| a * b).sum::<i64>())
        .collect();
    let y = (0..D)
        .map(|j| {
            let base: i64 = (0..D).map(|i| x[i] * w[i * D + j]).sum();
            base + SCALE
                * u.iter()
                    .zip(&adapter.b[j * R..][..R])
                    .map(|(a, b)| a * b)
                    .sum::<i64>()
        })
        .collect();
    (u, y)
}
