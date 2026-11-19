# gpu-lora-proof — Verifiable LoRA fine-tuning step (exact integer forward+backward)

**Status: COMPLETE — bounded spike executed on RTX 5060 Ti (SM120 Blackwell),
24/24 honest proofs accepted, 276/276 negative controls rejected, zero
failures.**

Can a GPU produce a compact certificate that an **exact integer LoRA
fine-tuning step** (forward *and* backward) was computed correctly, verifiable
on CPU in milliseconds, with tampering rejected? **Yes at d ≤ 256 (58–77 ms
pure-stdlib verification); verification stays sub-second to ~1 s at d = 1024
and scales as O(d²).** The certificate is a transparent Freivalds/Fiat–Shamir
proof over F_p (p = 2^61−1) with soundness error ≤ 9/p² < 2^−118 (target was
≤ 2^−63). This is **verifiable training**, not inference.

## What has actually run (on me4me, GPU 0 only, CUDA 13.3, driver 610.43.03)

| experiment | count | outcome |
|---|---|---|
| GPU prover: exact LoRA step + certificate (d∈{256,512,1024}, r∈{8,16}, b∈{1,16}, 2 seeds) | 24 configs × 7 launches (2 warmup + 5 measured) | all steps exact vs independent oracle |
| Independent pure-stdlib CPU verifier (YES/NO, exit code) | 24 honest proofs | **24/24 YES**, 138 exact checks each (3,312 total) |
| Exact rank-1 CPU oracle replay (different algorithm, no modular arithmetic) | 24 proofs | **12,434,208 elements, 0 mismatches** |
| Negative controls (11 tampers/config + 12 cross-seed stale replays) | 276 | **276/276 rejected** |
| Stdlib unittest self-test (CPU backend, honest + tampered fixtures) | 10 tests | 10/10 pass (local Python 3.14 and server 3.12) |

## Main result

- **Correctness:** every claimed output/gradient of the integer LoRA step is
  certified by 18 Freivalds checks (9 GEMMs × 2 Fiat–Shamir challenges) plus
  4 exact elementwise combination checks, 18 SHA-256 leaf commitments and 5
  trusted input pins. Honest proofs accept; coherent re-sealing tampers
  (attacker recomputes all hashes, challenges and responses) are rejected by
  the arithmetic itself.
- **GPU cost (kernel-only, CUDA events):** step 0.07–0.13 ms, certificate
  0.63–2.40 ms across all configs. Tensor cores (INT8 IMMA) used for 5 of 9
  GEMMs incl. an exact base-128 limb split of W; int64 SIMT where operands
  exceed int8 (see METHODS.md path table).
- **CPU verification (pure Python stdlib, median):** 58–77 ms (d=256),
  230–259 ms (d=512), 926–1021 ms (d=1024).
- **Proof size:** 1.60 MB (d=256, b=1) … 25.01 MB (d=1024, b=16) — transparent,
  dominated by W and dW (d² int64 each). Not succinct; no zk claim.

Full measured tables: [RESULTS.md](RESULTS.md). Math contract and soundness:
[METHODS.md](METHODS.md). Conclusions and next decisions:
[FINAL_REPORT.md](FINAL_REPORT.md). Raw data: `results/results.json`
(environment, source SHA-256s, per-run timings, hashes, control reasons).

## Scope limitations (explicit)

- **Synthetic model**: seeded random integer tensors, one LoRA step on a
  single d×d layer with a rank-r adapter. No real network, no dataset.
- **Integer arithmetic contract only** (int8 activations/grads, int16-range
  weights, fixed scale s=8, proven int64/int32 bounds). **No claim about
  float training** convergence or about full-model zkML.
- Transparent proof (MB-scale), not a succinct argument; verifier work is
  O(d²) modular dot products, not O(1).
- Soundness is computational (Fiat–Shamir in the random-oracle model) with
  the stated ≤ 2^−118 bound; the elementwise combines and input pins are
  exact/deterministic.
- Single GPU (RTX 5060 Ti, SM120), single step; no multi-step chain, no
  optimizer-state certification, no distributed setting.

## Related work (cited, not re-run)

- `requant-network/gpu-verifiable-inference` — inference-only proofs, fused
  SM120 kernels, CPU verifier with YES/NO. This study extends the paradigm
  from inference to a **training step** (forward + backward + weight/adapter
  gradients).
- `requant-network/Abacus` — Freivalds matmul PoW, Fiat–Shamir challenges,
  soundness ≤ 2^−63. We reuse the Freivalds/Fiat–Shamir technique with a
  larger Mersenne field (2^61−1) and two challenges per GEMM to reach
  ≤ 2^−118 over the whole 9-GEMM step.

## Layout / reproduce

```
contract.py            serialization + Fiat–Shamir contract (shared spec only)
cuda/lora_kernels.cu   SM120 kernels: INT8 IMMA (WMMA) + int64 SIMT + F_p matvecs
prover.py              GPU/CPU prover, commitments, challenges, proof JSON
verifier.py            independent pure-stdlib CPU verifier (YES exit 0 / NO exit 1)
oracle.py              independent exact rank-1 integer replay
tamper.py              12 negative controls (incl. coherent re-sealing attacker)
run.py                 full sweep runner (env gate, timings, controls, results.json)
experiments/make_report.py  regenerates RESULTS.md from results.json
tests/                 stdlib unittest self-test + committed tiny fixtures
results/results.json   measured raw data with hashes
```

Build (server, CUDA 13.3): `/usr/local/cuda/bin/nvcc -std=c++17 -O3
-arch=sm_120 -shared -Xcompiler -fPIC cuda/lora_kernels.cu -o liblora.so`

Run sweep (GPU 0 + shared lock): `CUDA_VISIBLE_DEVICES=0
flock /tmp/requant-new-studies-gpu.lock -c 'python3 run.py --lib ./liblora.so'`

Tests (no GPU): `python3 -m unittest discover -s tests -v`

Verify a proof: `python3 verifier.py <proof.json> <expected.json>` → `YES`/`NO`.