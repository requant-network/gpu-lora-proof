# Qwen Groth16 prover with the H term on CUDA

This public Qwen head-release fixture uses the real G1 H query.
The experiment measures the entire proof construction with the
CUDA implementation placed inside the prover. Both paths use
the same uncompressed, checked proving key, public manifest,
adapter, circuit and canonical 128-byte proof format.
[CPU key-format measurements](PROVING_KEY_FORMAT.md) ·
[CUDA H kernel and standalone measurements](https://github.com/requant-network/gpu-msm-tensorcore/blob/13ed7e91f16d471c6cd5824b91d4742bb128001c/FULL_MSM.md)

## Method

The CUDA path builds the ordinary R1CS assignment and Arkworks
QAP witness, calls the CUDA H multiplier, and inserts the
validated G1 point in C. It uses the upstream Arkworks proof
assembler for A, B and the remaining terms of C. For every CUDA
proof the command checks the original verification-key pin and
verifies the completed proof before writing it.

Measure one warm-up pair and six subsequent pairs in a single
process with the proving key resident. Alternate which path runs
first. For each pair use fresh randomness shared by its two
provers, and require equality of the canonical proof bytes.
Measure three additional pairs of fresh processes with warm
filesystem caches and independent proof randomness. In those
processes the CUDA path includes context initialization and its
local verification. The CPU path uses the stock proving command.
Measure transparent verification against the original bundle on
the same two physical CPU cores.

Independently check the saved proofs with the previous CPU-only
verifier in a read-only container with networking disabled.
Supply a well-formed wrong H point through a test library and
require the CUDA path to refuse publication.

## Results

| Scope | Stock CPU Groth16 | Groth16 with CUDA H | Wall-time reduction |
| --- | ---: | ---: | ---: |
| RTX-feature Qwen, resident key, complete prover | 2.218 s | 1.556 s | 29.8% |
| CPU-feature Qwen, resident key, complete prover | 1.270 s | 0.943 s | 25.8% |
| RTX-feature Qwen, fresh process and warm file cache | 10.766 s | 10.636 s | 1.2% |

The RTX-feature H query has 262,143 pairs. With the key resident
and CUDA initialized, the H call took a median 236 ms including
device allocation, transfers and affine output. Median constraint
synthesis and the QAP witness remained about 649 ms and 383 ms.
The CPU-feature profile has 131,071 pairs and exercised the
same paired resident protocol. CUDA ran on the RTX 5060 Ti;
the stock prover used two physical cores on the LAN Ryzen 5 5500.

In the fresh RTX-feature processes, checked key loading still
took 8.0–8.2 s. The CUDA path includes a new GPU context and
about 51 ms of proof verification. Median peak process RSS was
635 MiB for CPU and 767 MiB for CUDA. The 1.2% difference is
specific to the three measured cold process pairs. A service
with a key already loaded can use the resident result for its
latency estimate.

The transparent CPU builder recovered the archived certificate
byte for byte on all three repeats, taking a median 343 ms
from process start. Its separate verification took 104 ms
and recovered the same adapter. The compact verifier took about
76 ms inside the CPU command. The compact delivery with the cached
verification key is 8,325 bytes versus 211,090 for the transparent
package. First delivery with the compact key is 123,373 bytes.
The public fixture favors the transparent path for compute time
and the compact path for transfer size.

All fourteen paired CPU and CUDA proofs matched byte for byte.
The previous CPU-only verifier accepted those proofs and all six
fresh CPU and CUDA proofs in the isolated container. It refused a
modified adapter. The wrong on-curve H control reached Groth16
verification and was rejected before a proof file was written.
The default build reproduced the original H export, and seven
Rust and fourteen Python tests passed.
[RTX profiles and proof digests](results/cuda-h-prover/report.json) ·
[CPU-feature resident profiles](results/cuda-h-prover/cpu-feature.json) ·
[RTX offline verification](results/cuda-h-prover/rtx-offline.txt) ·
[CPU-feature offline verification](results/cuda-h-prover/cpu-offline.txt) ·
[Transparent rebuild](results/cuda-h-prover/transparent-build.json) ·
[Wrong-point control](results/cuda-h-prover/wrong-result.txt) ·
[Modified adapter](results/cuda-h-prover/edited-adapter.txt) ·
[Retained proofs](results/cuda-h-prover/proofs)

Qualification covers two public statements on one lab RTX 5060 Ti.
Private-witness deployment, G2 acceleration and measurements on
other device classes remain open.
The [trusted private-order benchmark](TRUSTED_CUDA_PRIVATE_PROVER.md)
uses the same H implementation with one checked key across three
synthetic private orders.

## Reproduce

Build the pinned CUDA library from the full-group study. Build the
experimental binary against that library on Linux. The usual
Cargo build produces a CPU-only binary for verification.

```sh
export MSM_LIB_DIR=/absolute/path/to/compiled/full-msm
cargo rustc --release --locked --manifest-path compact-round/Cargo.toml \
    --features cuda-h -- -L "native=$MSM_LIB_DIR"
export CUDA_VISIBLE_DEVICES="$LAB_RTX_UUID"
export LD_LIBRARY_PATH="$MSM_LIB_DIR"
flock /tmp/requant-new-studies-gpu.lock taskset -c 0,1 python3 -B \
    compact-round/cuda_h_trial.py compact-round/target/release/compact-lora-round \
    /absolute/path/to/cpu-verifier /absolute/path/to/uncompressed.key NEW_OUTPUT
```

Use the RTX-feature proving key and the verification key from
the same setup. Select two distinct physical CPU cores.
The alternate backend is intended for this public statement.
[Build and dependency pins](results/cuda-h-prover/environment.txt) ·
[Rust tests](results/cuda-h-prover/rust-tests.txt) ·
[Python tests](results/cuda-h-prover/python-tests.txt)
