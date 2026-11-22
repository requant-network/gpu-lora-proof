# CUDA H in the private three-order Groth16 benchmark

The established [three-template circuit](PRIVATE_TEMPLATE_TRIPLE.md)
produces one Groth16 proof for each of three private Qwen release orders.
The CPU and CUDA paths use the same checked compressed proving key and
the same committed orders. The CUDA path evaluates the G1 H term on
an operator-controlled RTX 5060 Ti. Constraint synthesis, QAP witness
construction, the remainder of proof assembly and verification run on
the CPU.

The benchmark checks all three private openings and circuit assignments
before loading the key. It prepares the verification key once, runs one
matched warm-up and evaluates one CPU and one CUDA proof for each order
with the key resident. Each matched pair uses fresh shared randomness
and requires identical canonical proof bytes. The paired execution order
alternates. The CUDA proof must also pass full Groth16 verification before
the command can write the three-proof file.

The separate CPU-only arbiter receives the shared verification key and
each order's public commitments, proof and issued order pin. It verifies
the submitted proofs in read-only, network-disabled containers. The
customer-side opening checks use the private adapter and its opening
locally. The CUDA process and GPU can observe the private H coefficients.
The kernel's execution depends on scalar digits. This fixture uses
synthetic private orders and a trusted accelerator under the same operator.

## Results

| Scope | CPU | CUDA H |
| --- | ---: | ---: |
| Median full proof construction with resident key | 5.480 s | 4.343 s |
| Sum over the three measured orders | 16.445 s | 13.014 s |

The median fell by 20.8%. Each H query contained 524,287 pairs
and the CUDA call took a median 463 ms, including allocation and transfer.
Checked loading of the 91,614,320-byte compressed key took 47.00 s.
The paired command took 90.43 s including key loading, prechecks,
warm-up and both paths. Its peak process RSS was 1.81 GiB. The RTX board
memory probe recorded 279 MiB at baseline and 1,996 MiB at peak
over the 200 ms samples. The benchmark used two pinned CPU cores on
the LAN Ryzen 5 5500 and the local RTX 5060 Ti.

All four pairs, including the discarded warm-up, matched byte for byte.
The retained three proofs had 128 bytes each. Their two commitments
per order and the shared verification key brought the public verification
material to 904 bytes. All three passed the prior CPU-only verifier
independently. The CUDA benchmark rejected a valid on-curve wrong point
before publishing a batch. A third order incompatible with its
chosen template failed before opening the key, and the arbiter refused a
proof from another order. The existing public CUDA prover still produced
a proof accepted by the CPU verifier. Seven Rust and fourteen Python
tests passed.

[Proofs and paired timings](results/cuda-h-private/public/report.json) ·
[Board memory](results/cuda-h-private/gpu-memory.json) ·
[Build pins](results/cuda-h-private/environment.txt) ·
[Wrong point](results/cuda-h-private/wrong-private-cuda-h.txt) ·
[Invalid third order](results/cuda-h-private/invalid-third-cuda-order.txt) ·
[Public-path control](results/cuda-h-private/public-regression-verify.txt).

## Reproduce

Build the pinned [G1 CUDA library and prover](CUDA_H_PROVER.md#reproduce).
Use the same Cargo toolchain, an ordinary CPU build as the arbiter, the
selected lab RTX and two pinned physical CPU cores. The script creates
a disposable checked key and temporary private inputs inside a fresh
directory. The optional wrong-point test uses the same test library as
the public-prover control.

```sh
CUDA_VISIBLE_DEVICES="$LAB_RTX_UUID" LD_LIBRARY_PATH="$MSM_LIB_DIR" \
flock /tmp/requant-new-studies-gpu.lock taskset -c 0,1 python3 -B \
    compact-round/private_pair_trial.py /path/to/cuda-prover NEW_OUTPUT \
    --cuda-h-cpu-verifier /path/to/ordinary-cpu-verifier \
    --cuda-h-wrong-library /path/to/wrong-point-library
```

Use `NEW_OUTPUT/public` as the retained verification evidence. The
operator-side benchmark logs record the timing and build environment.
The retained byte count covers the verification key, commitments and
proof files.
