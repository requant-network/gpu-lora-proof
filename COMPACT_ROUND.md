# Compact proof comparison for a LoRA round

The November 21 experiment compared three ways to check the same final adapter
on each retained synthetic LoRA manifest. On the lab Ryzen 5 5500, replay took
2.30–2.37 ms. The 128-byte Groth16 proof took 103–107 ms to verify and about
6.5 seconds to produce from its saved proving key. Direct replay remains the
cheapest check for these public inputs. [Raw measurements](results/compact-round/report.json).

The later [private-round experiment](PRIVATE_ROUND.md) evaluates a different
disclosure contract with the order and adapter behind commitments.
The [compact v2 follow-up](COMPACT_PREFIX.md) proves the accepted
real Qwen CPU and RTX head transitions, including prefix selection
and the held-out accuracy gate.

## Assertion and trust

The verifier pins the exact manifest bytes and independently receives the
expected checking-key SHA-256. The circuit takes the manifest pin, frozen W,
initial A and B, four ordered training slots, held-out features and targets,
and the claimed final A and B as public inputs. It checks `XW + 8(XAᵀ)Bᵀ`,
clips the residual to ±127, computes the selected gradient in each slot,
sums the two gradients, and applies bounded sign-SGD. Phase zero updates B.
Phase one uses that B to update A. The final adapter must reproduce every
pinned held-out target. This circuit admits the zero-SSE gate of both retained
manifests.

Both verification paths enforce the same final-adapter calculation. The
transparent certificate additionally preserves the ordered worker records and
the other single-step output matrices. Groth16 answers the arithmetic question
without establishing which operator performed the work. The inputs and output
are public. The disposable setup and the two trusted pins are part of the trust
model. The circuit has 78,850 constraints, 77,058 witness variables and 3,586
public scalars. [Round data contract](ROUND.md) · [Arkworks Groth16 implementation](https://github.com/arkworks-rs/groth16).

Public values are range-checked before allocation. For the allowed input
ranges, predictions and residuals have magnitude below `2^30`, and the
summed gradients have magnitude below `2^32`. The signed-integer gadgets
constrain the decomposed value back to the BN254 scalar field, enforce the
clipped residual and bound the optimizer step.

## Measurements

Three fresh-process trials per cohort ran on Linux using Rust 1.98.1, Python
3.12 and two Rayon threads. The transparent builder used two CPU contributors
and reproduced the byte-identical retained round bundle. Times are process-wall
medians per cohort. Peak RSS is the largest value reported by GNU time, in KiB.

| Check | Seed 20261121 | Seed 90817 | Peak RSS |
| --- | ---: | ---: | ---: |
| Direct integer replay | 2.30 ms | 2.37 ms | 3,456 KiB |
| Transparent certificate assembly | 312 ms | 305 ms | 21,376 KiB |
| Transparent certificate verification | 52.2 ms | 52.0 ms | 19,088 KiB |
| Groth16 proof generation, including key load | 6.48 s | 6.50 s | 246,576 KiB |
| Groth16 verification, including key and proof load | 107 ms | 103 ms | 4,480 KiB |

One setup took 2.22 seconds and 185,340 KiB peak RSS. It produced a 17,097,808-byte
proving key and a 115,016-byte verification key. Deserializing and validating
the proving key took 5.39–5.40 seconds. After that, proving took 1.07 seconds.
The four MSM-containing accumulator spans took 0.549–0.553 seconds, about half
of core proving and under 9% of the cold prover process. Eliminating those
spans would leave replay faster by orders of magnitude for this case. This
profile gives the reason to keep the current CPU prover for the tested fixture.

| Transfer to verifier | Seed 20261121 | Seed 90817 |
| --- | ---: | ---: |
| Manifest + transparent bundle | 211,401 bytes | 211,351 bytes |
| Manifest + final adapter + compact proof, with cached checking key | 8,647 bytes | 8,592 bytes |
| Same compact artifacts with first verification key download | 123,663 bytes | 123,608 bytes |

Both cohorts rejected a changed W entry, a changed manifest identity with the
same numeric inputs, a changed claimed adapter, a wrong manifest pin, a wrong
verification-key pin and extra proof bytes. Both circuit boundary tests passed.
The earlier round verifier recovered the same retained adapter in every
transparent check. Both retained proofs also passed verification from a separate
directory containing the public fixtures, the proofs and the verification key.
[Evidence and controls](results/compact-round/report.json) ·
[Execution log](results/compact-round/checks.log).

## Verify or reproduce

Build the locked CPU verifier and verify the first retained proof. The keys
and the proof are compressed Arkworks encodings. The verification key, both
128-byte proofs and the previously pinned manifests and adapters suffice for
verification. [Retained verification key](results/compact-round/verifying.key) ·
[Seed 20261121 proof](results/compact-round/20261121.proof) ·
[Seed 90817 proof](results/compact-round/90817.proof).

```sh
cargo build --manifest-path compact-round/Cargo.toml --release --locked
cargo test --manifest-path compact-round/Cargo.toml --release --locked
compact-round/target/release/compact-lora-round verify \
  results/round-v1/20261121/manifest.json \
  c645e19ef587a2b582ceba48719de32032e5584d33eacdbcf33b0db39421fe65 \
  results/round-v1/20261121/adapter.json results/compact-round/verifying.key \
  ec74c5322c33e420ee3343b2181adf6869fe4fff2af2379ae9101fe3b21c7833 \
  results/compact-round/20261121.proof
python3 -B compact-round/benchmark.py compact-round/target/release/compact-lora-round ./new-comparison
```

The benchmark generates a fresh proving key locally and retains the new
verification key pin with its results.
