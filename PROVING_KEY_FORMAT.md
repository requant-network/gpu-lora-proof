# Checked proving-key formats for the Qwen release proof

The [public Qwen release proof](COMPACT_PREFIX.md) spends most of its
cold-process wall time loading the BN254 Groth16 proving key. This
trial compares the checked compressed and uncompressed Arkworks
encodings of the same key for each retained feature profile. It also
measures how much wall time lies inside the four MSM-containing prover
phases.

## Protocol

Use the existing disposable proving keys for the RTX- and CPU-feature
statements, with the original manifest, adapter and verification key
for each. Both proof computations run on the LAN Ryzen 5 5500. Keep
the compressed command unchanged and use the dedicated uncompressed
command for the alternate key. Both use Arkworks `Validate::Yes` and
require complete consumption of the key bytes. The active size limit
is 128 MiB per proving key in either encoding.

Convert each key once before timing. Run four paired fresh processes
per profile with two Rayon threads pinned to the same two physical
cores. Alternate the encoding order between pairs. File cache state
is warm after conversion. Record both end-to-end prover wall time and
the checked key load. Setup, conversion and verification have separate
clocks. Treat the sum of `Compute A`, `Compute B in G1`, `Compute B in G2`
and `Compute C` as an optimistic bound on time a GPU MSM could remove.
These phases also include CPU work outside MSM.

Verify each new 128-byte proof with the original pinned verification key
and original statement. Verify the retained historical proofs with the
same verifier. Change the first uncompressed point to an off-curve
encoding and separately append one byte to a valid key. Require both
inputs to fail before writing a proof. Repeat the new proof checks
inside a read-only container with networking disabled.

## Results

| Median of four runs per encoding | RTX features, compressed | RTX features, uncompressed | CPU features, compressed | CPU features, uncompressed |
| --- | ---: | ---: | ---: | ---: |
| Full prover process | 18.11 s | 10.80 s | 14.53 s | 8.79 s |
| Checked key loading | 15.33 s | 8.03 s | 12.83 s | 7.08 s |
| Proof construction after loading | 2.42 s | 2.41 s | 1.40 s | 1.40 s |
| Sum of MSM-containing phases | 1.146 s | 1.137 s | 0.613 s | 0.612 s |
| Peak RSS | 632.7 MiB | 632.8 MiB | 430.8 MiB | 430.7 MiB |
| Proving key | 32,074,992 bytes | 64,149,936 bytes | 24,470,768 bytes | 48,941,488 bytes |

Wall time decreased by **40.4%** on the RTX-feature statement and
**39.5%** on the CPU-feature statement. Conversion cost 15.62 s
and 12.99 s respectively, including the checked load of the
source. At the measured medians, a local key converted once
recovers that time on the third proof. Storage approximately doubles.

With compressed keys, the four MSM-containing phases use
6.3% and 4.2% of the process wall time. With uncompressed keys they
use 10.5% and 7.0%. After the key is loaded, the same phases
account for about 47% and 44% of proof-creation time. These ratios
put the prospective GPU experiment on the resident-key path.
Actual acceleration would depend on group arithmetic, transfers
and circuit-specific scalar distributions. The related study
measured a separate linear sub-step.
[Coordinate accumulation study](https://github.com/requant-network/gpu-msm-tensorcore/blob/fc3e449becc2b46d4497e9709b3871bdf447f8b3/README.md)

All sixteen new proofs and both historical public proofs verified
in a read-only network-disabled container using the original
verifier keys. The modified adapter was rejected against an
uncompressed-key proof. Both invalid-key cases failed before
creating a proof. The existing private three-proof offline
verification, seven Rust tests and fourteen Python tests passed.
[Measurements and key digests](results/proving-key-format/report.json) ·
[Proofs](results/proving-key-format/proofs) ·
[RTX offline log](results/proving-key-format/rtx-offline-review.txt) ·
[CPU offline log](results/proving-key-format/cpu-offline-review.txt) ·
[Modified-adapter control](results/proving-key-format/changed-adapter.txt) ·
[Private proof regression](results/proving-key-format/private-regression.txt)

The alternate encoding applies to the proving key in this research
command. Both proof and verification-key formats remain compressed.
The prior private triple candidate has a 91,614,320-byte compressed
proving key and exceeds the 128 MiB limit when uncompressed.
The measurements used fresh prover processes with warm file caches.
GPU MSM and disk-cold proof runs remain open.

## Reproduce

Build the pinned binary and provide the original disposable compressed
keys for the two profiles. Choose two distinct physical CPU cores.
The fixture saves the generated proving keys locally in the owner-only
trial directory. The published evidence uses proof bytes, timings and
digests.

```sh
cargo build --release --locked --manifest-path compact-round/Cargo.toml
taskset -c 0,1 python3 -B compact-round/key_format_trial.py \
    compact-round/target/release/compact-lora-round RTX_COMPRESSED_KEY CPU_COMPRESSED_KEY NEW_OUTPUT
```

[Toolchain and source pins](results/proving-key-format/environment.txt) ·
[Rust tests](results/proving-key-format/rust-tests.txt) ·
[Python tests](results/proving-key-format/python-tests.txt)
