# FINAL REPORT — gpu-lora-proof

**Study:** Verifiable LoRA fine-tuning step: exact integer forward+backward
with Freivalds/Fiat–Shamir certificates and an independent CPU verifier.
**Type:** bounded spike (one core experiment + scale/timing sweep).
**Executed:** 2026-11-19 on me4me, RTX 5060 Ti (GPU 0, SM120, cc 12.0),
CUDA 13.3, driver 610.43.03, Python 3.12.3 stdlib. All numbers measured;
raw data in `results/results.json` with source SHA-256s.

## Question and answer

*Can a GPU produce a compact certificate that an exact integer LoRA
forward+backward step was computed correctly, verifiable on CPU in
milliseconds, with tampering rejected?*

**Yes, with stated qualifications.** A transparent Freivalds/Fiat–Shamir
certificate over F_{2⁶¹−1} (soundness ≤ 9/p² < 2⁻¹¹⁸) certifies all nine
GEMMs and four exact combines of one integer LoRA training step. An
independent pure-stdlib CPU verifier accepts every honest proof (24/24,
YES/exit 0) and rejects every tamper we constructed (276/276), including
coherent attackers that re-seal all hashes, challenges and responses.
Verification is **58–77 ms at d=256** (milliseconds, as asked), ~0.25 s at
d=512 and ~1 s at d=1024, scaling O(d²). "Compact" holds only relative to
recomputation-with-trust: proofs are 1.6–25 MB (transparent, not succinct).

## What ran (counts)

- 24 configurations: d ∈ {256, 512, 1024} × r ∈ {8, 16} × batch ∈ {1, 16} ×
  seeds {20261119, 987654321}; 7 prover launches each (2 warmup + 5 measured).
- 24 honest proofs → independent verifier: **24/24 accepted**, 138 exact
  checks each (3,312 total): 18 leaf hashes, 5 input pins, 18 range checks,
  18 Freivalds identities, 54 response bindings, 4 elementwise combines.
- Exact rank-1 integer oracle (third code path): **12,434,208 elements,
  0 mismatches**.
- Negative controls: **276/276 rejected** (11 kinds × 24 + 12 cross-seed
  stale replays).
- Stdlib self-test: 10/10 pass on local Python 3.14.9 and server 3.12.3;
  committed tiny fixture verifies YES on both.
- On-device preflight: `__int128` Mersenne reduction exact; INT8 WMMA tile
  0 mismatches vs CPU reference on sm_120.

## Key measured numbers (medians; full ranges in RESULTS.md)

| d | proof size | GPU step kernel | GPU cert kernel | CPU verify |
|---|---|---|---|---|
| 256 | 1.60–2.07 MB | 0.07–0.08 ms | 0.63–0.68 ms | 57.6–76.9 ms |
| 512 | 5.99–6.92 MB | 0.07–0.08 ms | 1.20–1.25 ms | 230.6–259.1 ms |
| 1024 | 23.16–25.01 MB | 0.11–0.13 ms | 2.30–2.40 ms | 926.2–1020.8 ms |

- Certificate overhead on GPU is tiny: kernel-only cert time is 0.6–2.4 ms
  vs 0.07–0.13 ms for the step itself (≈ 9–20× step cost, but both are
  sub-3 ms). Transfer-inclusive walls (5.8–138 ms step, 4–69 ms cert) are
  dominated by PCIe copies and allocations at this scale, not compute.
- CPU-side commitment + Fiat–Shamir: 4.1–78 ms (SHA-256 over the leaves).
- Verifier scaling ≈ 4× per doubling of d, consistent with the three d×d
  matvecs (H, P, dW checks) dominating; batch and r have minor effect.
- Tensor-core utilization: 5/9 GEMMs on INT8 IMMA (H, P via exact base-128
  limb split of W; U, Q, dW direct), 4/9 exact int64 SIMT (V, R, T8, T9 —
  operands exceed int8). All int32 accumulator bounds proven ≪ 2³¹; all
  int64 bounds < 2³⁴. Per-GEMM paths recorded in every proof/results JSON.

## Correctness / verification status

**PASS on every axis:** honest acceptance 24/24; exact oracle 24/24;
negative controls 276/276 rejected; self-tests 10/10; GPU identity gate
enforced in code (run aborts unless cc 12.0 RTX 5060 Ti). No failures, no
unexplained anomalies.

## Negative results and honest limitations

1. **Milliseconds only at small d.** Pure-stdlib verification at d=1024 is
   ~1 s, not ms. The O(d²) modular dot products are the bottleneck; a
   compiled verifier or batching would help, but that is outside this spike.
2. **Proofs are not succinct.** 1.6–25 MB, dominated by W and dW (d² int64
   each, base64). Commitment-only proofs (hashes + responses, leaves fetched
   on demand) would shrink the wire size but change the trust model; not
   explored here.
3. **Transfer overhead dominates GPU wall time** at these sizes; the
   kernel-only numbers show the certificate itself is cheap. A resident
   pipeline (weights already on device) would close most of the gap.
4. **Integer contract only.** No claim about float training, real models,
   optimizer state, multi-step chains, or zkML. The certificate proves one
   exact integer step under a fixed public contract; it says nothing about
   whether the step is a *good* training step.
5. **Stale-replay rejection surfaced via step_id** (seed embedded in the
   id), before input pins were reached — rejection is correct but the
   pin-based catch is demonstrated separately by alter_W/wrong_x controls.
6. Soundness is in the random-oracle model (SHA-256 Fiat–Shamir), standard
   for this technique (cf. Abacus); the 2⁻¹¹⁸ bound is a union bound over
   18 independent challenges.

## Relation to prior work (cited, not re-run)

- `requant-network/gpu-verifiable-inference`: inference-only certificates
  with fused SM120 kernels and a YES/NO CPU verifier. This study carries the
  paradigm into **training**: backward GEMMs, gradient leaves, and the
  extra failure modes (tampered gradients, wrong adapter grads) that
  inference proofs never face.
- `requant-network/Abacus`: Freivalds matmul PoW with Fiat–Shamir challenges
  and soundness ≤ 2⁻⁶³. We adopt the technique, enlarge the field to
  2⁶¹−1 and use two challenges per GEMM, reaching ≤ 2⁻¹¹⁸ for the whole
  9-GEMM step while keeping the verifier pure-stdlib.

## Next research decisions (recommended)

1. **Resident-weight prover**: keep W on device across steps; certify a
   *chain* of steps with a rolling transcript (each step's output hashes
   feed the next challenges). This is the natural path to verifiable
   fine-tuning runs rather than single steps.
2. **Compiled verifier** (C/Rust) to test whether d=1024 verification can
   reach the ms regime; the arithmetic is identical, only the interpreter
   overhead changes.
3. **Commitment-only wire format**: send hashes + responses; verifier
   fetches leaves lazily and checks only the challenged products — trades
   full transparency for size.
4. **Float bridge**: certified fixed-point emulation of a real LoRA step
   (quantize → integer step → certificate), bounding the quantization
   error separately; would connect this contract to actual training.
5. Optimizer-state certification (Adam moments are just more leaves/GEMMs;
   the contract extends mechanically).

## Reproducibility

Everything needed is in the repo: sources (`contract.py`, `prover.py`,
`verifier.py`, `oracle.py`, `tamper.py`, `run.py`, `cuda/lora_kernels.cu`),
committed tiny fixtures, stdlib tests, and `results/results.json` containing
environment, source SHA-256s, per-run timings, proof hashes and every
control's rejection reason. Server-side proof artifacts (248 MB) retained at
`me4me:~/requant-studies/gpu-lora-proof/results/proofs/` (gitignored).