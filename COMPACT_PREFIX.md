# Compact proof of the Qwen head release rule

November 21, 2026. The [v2 accepted head manifests](FEATURE_ORIGIN.md)
use a training-loss-selected step prefix and a held-out accuracy
floor. This comparison proves the same final adapter on the two
mixed-vocabulary feature profiles using Groth16 over BN254.
[Arkworks 0.5.0 implementation](https://docs.rs/ark-groth16/0.5.0/ark_groth16/).

## Statement

For each of the two phases, the circuit recomputes the eight forward
passes and clipped gradients, combines the two slots, and verifies the
full clipped sign step. The setup fixes the ordered list of active
coordinates and the sixteen expected label positions. The circuit checks
that each active coordinate changes, each unlisted coordinate stays
fixed, and the order follows descending absolute gradient with ascending
index for ties. It computes the unclipped squared error for every prefix,
including the empty prefix, and enforces the earliest minimum. The
second phase uses the first phase's proved factor.

The circuit recomputes the sixteen held-out predictions from the
resulting factors. It enforces the manifest's squared-error ceiling
and all sixteen correct labels, with ties resolved at the first
maximum. It also verifies each pinned label is the unique target
maximum. The host parser validates manifest bytes, public shapes
and bounds and separation between train and held-out rows. Integer
differences used by the minimum and error comparisons fit below
`2^68` under these bounds.

Both trials use disposable setup keys. The trusted key digest selects
the circuit, including its fixed ordering and label positions. The
original manifests and returned adapter values are public. The verifier
checks the adapter calculation given the pinned feature manifest.
Feature extraction and worker identities retain their separate checks.

## Results

| Measurement | RTX feature profile | CPU feature profile |
| --- | ---: | ---: |
| Active coordinates in B and A | 8 and 64 | 8 and 32 |
| Constraints | 147,125 | 125,749 |
| Setup | 4.851 s | 3.696 s |
| Proving key | 32,074,992 bytes | 24,470,768 bytes |
| Proof generation including key load and circuit check | 17.941 s | 14.428 s |
| Proof | 128 bytes | 128 bytes |
| Compact verification inside command, median | 75.7 ms | 76.5 ms |
| Transparent verification inside command, median | 64.9 ms | 41.0 ms |
| Transparent package | 211,090 bytes | 211,106 bytes |
| Compact package with cached key | 8,325 bytes | 8,334 bytes |
| Compact package including first verification key download | 123,373 bytes | 123,382 bytes |

Both trials used the same Linux Ryzen 5 5500 server and two Rayon
threads. The feature-profile labels refer to the source of the pinned
Qwen features. Both proof systems were measured on CPU. Each
verification median uses three fresh processes. Externally measured
process wall medians, rounded to centiseconds, were 0.07 s for each
compact profile and 0.10 s and 0.07 s for the transparent RTX and CPU
profiles. Interpreter startup affects that comparison. The proof path
reduced bytes substantially and incurred a new setup and prover cost.
[RTX report](results/compact-prefix/report.json) ·
[CPU report](results/compact-prefix/cpu/report.json).

The verifier accepted the expected public RTX and CPU adapter
statements in a directory with only the manifest, adapter, proof
and key. Eight controls rejected missing and reordered coordinates,
an insufficient error ceiling, a failed accuracy target, a changed
public adapter, changed manifest identity, stray proof bytes and a
wrong verification-key pin. The RTX key and proof failed verification against the CPU manifest and adapter.
The two earlier public and private v1 proofs still verified.
Seven Rust tests and fourteen Python tests passed.

The [proving-key format trial](PROVING_KEY_FORMAT.md) reuses these
two statements and profiles the checked key load and MSM phases.

## Verify or reproduce

The retained RTX and CPU packages contain separate verification keys
and proofs. The verifier uses the original manifest and adapter with
their published key and manifest pins.

```sh
cargo build --manifest-path compact-round/Cargo.toml --release --locked
bin=compact-round/target/release/compact-lora-round
rtx=results/text-head-v2/mixed-vocabulary
rtx_pin=d200e82458d79b44fc577480ba33b1e7fe79b6e935328bd7812cfdd05763c34a
"$bin" prefix-verify "$rtx/manifest.json" "$rtx_pin" "$rtx/adapter.json" \
  results/compact-prefix/verifying.key ef6ff88719ce916f2fc86a70e79de8e90470d4b0e534c5782ae9dbde8d8dd38d \
  results/compact-prefix/round.proof

cpu=results/text-head-origin/cpu
cpu_pin=ccb4d88429a4e2fce34b1f1a3d7cd88f1755f3ab6aaf9d91262ad0a52eb54263
"$bin" prefix-verify "$cpu/manifest.json" "$cpu_pin" "$cpu/adapter.json" \
  results/compact-prefix/cpu/verifying.key 082d83afa36e7d5867f93bc6e5d98a86bad2ac5c742cf0b84b584245b84970cd \
  results/compact-prefix/cpu/round.proof
```

For a fresh key and proof, derive the setup shape from the transparent
verified bundle. The shape needs to match the circuit proof of its
ordering. Keep the new public key pin from the setup output.

```sh
python3 -B compact-round/prefix_shape.py "$rtx/manifest.json" "$rtx/bundle.json" new-shape.json \
  --manifest-sha256 "$rtx_pin"
"$bin" prefix-check new-shape.json "$rtx/manifest.json" "$rtx_pin" "$rtx/adapter.json"
"$bin" prefix-setup new-shape.json new-keys
"$bin" prefix-prove new-shape.json "$rtx/manifest.json" "$rtx_pin" "$rtx/adapter.json" \
  new-keys/proving.key new-round.proof
```

## Export the G1 H query for a standalone MSM comparison

`prefix-export-h-msm` uses the same public statement and Arkworks
R1CS-to-QAP reduction as the prover. It checks the compressed proving
key's embedded verification-key digest and exports the actual G1
bases and scalar coefficients of the H term. The retained RTX-feature
key has 262,143 bases. Its witness map has one additional coefficient,
which the exporter checks is zero.

```sh
"$bin" prefix-export-h-msm new-shape.json "$rtx/manifest.json" "$rtx_pin" \
  "$rtx/adapter.json" new-keys/proving.key NEW_VERIFICATION_KEY_SHA256 qwen-h-msm.bin
```

Supply the digest reported by the setup that generated the proving key.
The file contains 64 bytes per affine base followed by 32 bytes per
scalar. Coordinates and scalars are unsigned little-endian integers.
An all-zero point encodes infinity. The command is a fixture export for
public statements. The resulting wire exposes the QAP coefficients.

Two exports reproduced all 25,165,728 bytes from the same checked key.
A different verification-key pin was refused before creating an output
file. The historical RTX proof still verified and seven Rust tests passed.
[Export digests and checks](results/compact-prefix/h-msm/lora-export-checks.txt) ·
[Rust tests](results/compact-prefix/h-msm/lora-tests.txt)
