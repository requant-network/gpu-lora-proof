# Public Qwen adapter delivery and verification

November 23, 2026. The same two [pinned Qwen v2 feature manifests](FEATURE_ORIGIN.md)
and returned adapters were checked by direct deterministic replay, a transparent
certificate, and Groth16. Direct replay recomputes the public adapter update and
the held-out release gate. The transparent certificate also retains the worker
contributions, step matrices and round root. Groth16 checks the public adapter
statement and release gate using a separately pinned verification key. These
checks start at the published feature vectors.

All packages include the same manifest and adapter for a given profile. The
transparent package also carries the certificate. The compact packages carry a
128-byte proof and include the public verification key on first delivery.
The two profile names refer to the source of the pinned features. All
measurements below used CPU.

## Results

| Delivery | RTX-feature bytes, raw / gzip | CPU-feature bytes, raw / gzip | Xeon receiver median, RTX / CPU |
| --- | ---: | ---: | ---: |
| Manifest and adapter for direct replay | 8,197 / 1,393 | 8,206 / 1,393 | 176.2 / 136.6 ms |
| Transparent certificate | 211,618 / 22,862 | 211,633 / 23,201 | 162.4 / 122.8 ms |
| Groth16 with cached key | 8,325 / 1,637 | 8,334 / 1,631 | 124.1 / 125.2 ms |
| Groth16 with first key delivery | 123,373 / 117,150 | 123,382 / 117,153 | 124.6 / 125.0 ms |

Raw bytes are the sum of the file sizes. Gzip bytes include the deterministic
tar headers. Receiver wall time includes unpacking and a fresh verifier process,
with four rotated measurements per form. Setup and producer work were measured
separately.

| Additional producer work | RTX features | CPU features |
| --- | ---: | ---: |
| Fresh Groth16 setup | 7.620 s | 5.789 s |
| Fresh proof generation including key load | 29.893 s | 23.870 s |
| Prover peak RSS | 647,868 KiB | 441,040 KiB |
| Transparent certificate assembly | 0.494 s | 0.454 s |

The measured fresh transactions start with an already known adapter.
Direct delivery needs only the two public files. With the key cached,
Groth16 used 238–244 more compressed bytes and measured 11–52 ms less
receiver wall time. Proof construction added 23.9–29.9 seconds before
delivery. A cached compact proof saved about 93% of the compressed bytes
compared with carrying the transparent records. Retain the transparent
certificate when a reviewer needs those records. Use direct replay for
this measured public statement when the decision is based on the cost
of a fresh delivery and check. The result covers these two public
feature manifests and this setup. Feature extraction, real training,
executor identity and private inputs have separate evidence.

The direct and transparent paths recovered identical adapter digests
and matched the 16/16 held-out labels. The reconstructed transparent
bundles matched the retained certificates byte for byte. Both direct
replay and Groth16 rejected an altered claimed adapter. An additional
LAN Ryzen 5 5500 run accepted both fresh proofs from the retained
first-delivery packages and independently recomputed the same two
adapter digests and held-out results.

## Reproduce

Build the CPU binary with the locked Cargo manifest. Run the benchmark
from the repository root with a new output directory. The Xeon
E5-2630 v4 trial used Linux, Python 3.12.3, GNU time, logical CPUs
0 and 1 and two Rayon threads in a network-isolated service. Both
machines used the same CPU binary, SHA-256
`c440e09c25dc6d417c5b58849e95c895426eb4088e190f048dd2b849ba7aa18e`.

```sh
cargo build --release --locked --manifest-path compact-round/Cargo.toml
taskset -c 0-1 python3 -B compact-round/public_handoff.py benchmark \
  compact-round/target/release/compact-lora-round /path/to/new/public-handoff
```

Use the [protocol](PUBLIC_HANDOFF_PROTOCOL.md),
[Xeon results and package hashes](results/public-handoff/report.json),
and [Ryzen review](results/public-handoff/ryzen-review.json).
The retained archive directories contain the delivered files for
[RTX features](results/public-handoff/rtx-features/) and
[CPU features](results/public-handoff/cpu-features/). Fresh disposable
setup produces new proof and verification-key bytes on rerun.
