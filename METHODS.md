# METHODS — gpu-lora-proof

Verifiable LoRA fine-tuning step: exact integer forward+backward with
Freivalds/Fiat–Shamir certificates and an independent CPU verifier.

## 1. Arithmetic contract (all integers, no float anywhere in claims)

One LoRA layer, one training step. Tensors (row-major, int64 storage):

| tensor | shape | range | role |
|---|---|---|---|
| x | b×d | int8, \|x\| ≤ 127 | input activations |
| W | d×d | \|W\| ≤ 1024 (int16-range) | frozen base weight |
| A | r×d | \|A\| ≤ 64 | LoRA down-projection |
| B | d×r | \|B\| ≤ 64 | LoRA up-projection |
| g | b×d | int8, \|g\| ≤ 127 | upstream loss gradient dL/dy |
| s | scalar | s = 8 | fixed LoRA scale |

**Forward** (y = x·W + (x·Aᵀ)·Bᵀ·s):

1. `H = x·W`            (b×d)
2. `U = x·Aᵀ`           (b×r)
3. `V = U·Bᵀ`           (b×d)
4. `y = H + s·V`        (b×d)

**Backward** (exact chain rule for this layout):

5. `P = g·Wᵀ`           (b×d)   — base-weight path of dX
6. `Q = g·B`            (b×r)   — dL/dV = g, then dL/dU = g·B
7. `R = Q·A`            (b×d)   — dL/dX through the adapter: (g·B)·A
8. `dX = P + s·R`       (b×d)
9. `dW = xᵀ·g`          (d×d)   — base-weight gradient
10. `T8 = Qᵀ·x`         (r×d); `dA = s·T8`  — adapter-down gradient
11. `T9 = gᵀ·U`         (d×r); `dB = s·T9`  — adapter-up gradient

Nine GEMMs total (H, U, V, P, Q, R, dW, T8, T9) plus four exact elementwise
combines (y, dX, dA, dB). All 18 leaves (5 inputs + 9 GEMM outputs + 4
combined outputs/grads) are committed.

### Exactness bounds (K·max₁·max₂ per GEMM)

Worst-case absolute values (d=1024, r=16, b=16):

| leaf | bound | value |
|---|---|---|
| H, P | d·127·1024 | 133,169,152 |
| U | d·127·64 | 8,323,072 |
| Q | d·127·64 | 8,323,072 |
| V | r·U·64 | 8,522,825,728 |
| R | r·Q·64 | 8,522,825,728 |
| y, dX | H + s·V | 68,315,774,976 |
| dW | b·127·127 | 258,064 |
| T8 | b·Q·127 | 16,912,347,136 |
| T9 | b·127·U | 16,912,347,136 |
| dA, dB | s·T8 / s·T9 | 135,298,777,088 |

Max leaf value ≈ 1.35×10¹¹ < 2⁴⁰ ≪ 2⁶³: **every intermediate fits int64
exactly**. All bounds are enforced by the verifier per leaf (range check),
so a prover cannot smuggle out-of-contract values.

### GPU execution path per GEMM (Blackwell directive)

All GPU work: RTX 5060 Ti only, GPU 0, `-arch=sm_120` (SM120 Blackwell).
INT8 IMMA via WMMA `mma.sync` 16×16×16, int32 accumulators, where operands
fit int8 with **proven int32 accumulation bounds**; exact int64 SIMT
otherwise. Recorded in every proof (`gemm_paths`) and results JSON:

| GEMM | path | justification / int32 bound |
|---|---|---|
| H = xW | **IMMA, base-128 limb split** | W = 128·W1 + W0 with \|W1\| ≤ 8, \|W0\| ≤ 64 (both int8). Two IMMA GEMMs, recombine H = 128·(xW1) + (xW0) in int64. Acc bounds: d·127·8 = 1,040,384 and d·127·64 = 8,323,072 ≪ 2³¹ |
| U = xAᵀ | **IMMA direct** | x, A int8; acc bound d·127·64 = 8,323,072 ≪ 2³¹ |
| V = UBᵀ | **int64 SIMT** | U up to 8.3×10⁶ exceeds int8; int64 bound r·U·64 < 2³⁴ |
| P = gWᵀ | **IMMA, base-128 limb split** | same split of W; same acc bounds as H |
| Q = gB | **IMMA direct** | g, B int8; acc bound d·127·64 ≪ 2³¹ |
| R = QA | **int64 SIMT** | Q exceeds int8 |
| dW = xᵀg | **IMMA direct** | x, g int8; acc bound b·127·127 = 258,064 ≪ 2³¹ |
| T8 = Qᵀx | **int64 SIMT** | Q exceeds int8; int64 bound b·Q·127 < 2³⁴ |
| T9 = gᵀU | **int64 SIMT** | U exceeds int8; int64 bound b·127·U < 2³⁴ |

Limb-split correctness: for \|W\| ≤ 1024, W1 = round(W/128) ∈ [−8, 8],
W0 = W − 128·W1 ∈ [−64, 64]; the identity H = 128·(xW1) + (xW0) is exact in
integers, and each partial product's int32 accumulator is bounded as above
(no overflow possible at any swept d, b). The split is recomputed on-device
from the committed W bytes.

The limb split and the WMMA INT8 path were validated on-device before the
study: (a) `unsigned __int128` multiply + mod (2⁶¹−1) smoke test returned the
exact expected residue; (b) a 16×16×16 INT8 WMMA tile matched an exact CPU
reference with **0 mismatches** on sm_120.

## 2. Certificate construction

**Field.** p = 2⁶¹−1 (Mersenne prime). Products of two field elements fit
`unsigned __int128`; reduction is shift/add folding (no division). Every
claimed leaf value satisfies \|v\| < 2⁴⁰ < p/2, so the integer→field map is
injective on the contract domain: a field identity failure implies an
integer identity failure.

**Commitments.** Each of the 18 leaves is hashed:
`SHA-256(name ‖ shape ‖ little-endian int64 data)`. The transcript is the
ordered concatenation of the 18 digests plus schema/step_id/params.

**Fiat–Shamir challenges.** For GEMM i (fixed order H,U,V,P,Q,R,dW,T8,T9)
and repetition j ∈ {0,1}:
`r = SHA-256(transcript ‖ i ‖ j ‖ block)` counter mode, 4 uint64 per block,
each reduced mod p. Two independent challenge vectors per GEMM (18 total,
lengths = output dimension of the GEMM).

**Freivalds certificate per GEMM.** For claim C = A·B (m×k · k×n) and
challenge r ∈ F_pⁿ the prover computes, on GPU:
- `u = B·r`  (k-vector, SIMT over F_p)
- `v = A·u`  (m-vector, SIMT over F_p)
- `w = C·r`  (m-vector, SIMT over F_p)

and includes (u, v, w) in the proof. If C ≠ A·B then for uniform r,
Pr[(A·B − C)·r = 0] ≤ 1/p (Schwartz–Zippel, degree 1). The verifier
recomputes u, v, w independently and checks v = w **and** that the proof's
vectors equal its own recomputation (response binding), so a prover cannot
substitute fabricated consistent-looking vectors without knowing r — which
is derived from the commitments it already sealed.

**Soundness.** 9 GEMMs × 2 independent challenges: union bound
≤ 9·(1/p)² = 9/p² ≈ 1.6×10⁻³⁶ < **2⁻¹¹⁸**, comfortably below the 2⁻⁶³
target. (Challenges are derived from a SHA-256 transcript — random-oracle
model, as in Abacus.) The four elementwise combines, the 18 leaf hashes, the
5 input pins and all range checks are exact/deterministic, adding no
soundness slack.

**Proof contents.** schema + step_id + params; 18 leaves (base64 int64) +
their SHA-256 hashes; 54 response triples (u, v, w); per-GEMM path codes;
prover timings. No secret material; fully transparent.

## 3. Independent CPU verifier (different code path)

`verifier.py` is pure Python stdlib and shares **only** the serialization /
Fiat–Shamir contract (`contract.py`) with the prover — no arithmetic code is
shared. Verification steps (138 checks per proof):

1. schema, step_id, params equal the trusted expected-inputs manifest (3);
2. all 18 leaves present, shapes correct, base64 decodes to int64 (18);
3. each leaf's SHA-256 equals the declared hash (18);
4. input leaves W, A, B, x, g pinned to trusted expected hashes (5);
5. every leaf value within its proven worst-case bound ⇒ \|v\| < p/2 (18);
6. recompute the Fiat–Shamir transcript and all 18 challenge vectors;
7. per GEMM per rep: recompute u = B·r, v = A·u, w = C·r with stdlib
   `sum(map(mul, ...))` modular dot products; check v = w (18) and bind the
   proof's u, v, w to the recomputation (54);
8. exact integer elementwise combines y = H+sV, dX = P+sR, dA = s·T8,
   dB = s·T9 over every element (4 checks).

Prints `YES` (exit 0) or `NO: <reason>` (exit 1).

**Second independent oracle.** `oracle.py` replays all nine GEMMs by rank-1
outer-product accumulation in exact Python integers (no modular arithmetic,
no shared GEMM code with prover or verifier) and compares every output leaf
elementwise. 12,434,208 elements checked across the sweep, 0 mismatches.

## 4. Negative controls (12 kinds)

Single-proof tampers, applied to honest GPU proofs; the six "coherent"
variants re-hash all leaves, re-derive all challenges and recompute all 54
response vectors for the tampered claims (maximally capable re-sealing
attacker):

| control | tamper | detection mechanism |
|---|---|---|
| flip_dX_coherent | flip one dX element, re-seal everything | exact combine dX |
| alter_W_coherent | change one W element, re-seal | trusted input pin W |
| wrong_y_coherent | perturb one y element, re-seal | exact combine y |
| wrong_dW_coherent | perturb one dW element, re-seal | Freivalds GEMM dW |
| flip_U_coherent | flip one U element, re-seal | Freivalds GEMM U |
| wrong_x_coherent | change one x element, re-seal | trusted input pin x |
| truncated_responses | drop response bytes | decode failure |
| truncated_leaf | truncate W leaf data | decode failure |
| wrong_step_id | different step id | step_id check |
| wrong_scale | s = 9 instead of 8 | params check |
| tamper_response_v | corrupt one response vector | response binding |
| stale_replay_cross_seed | present seed-B proof against seed-A manifest | step_id / pins |

All 276 instances (11×24 + 12) rejected. Note: a coherent tamper of an
*intermediate* GEMM leaf that is never combined elementwise (e.g. dW) is
caught only by Freivalds — which is exactly what the certificate is for.

## 5. Timing boundaries

- **step kernel / cert kernel**: CUDA-event span around compute kernels only
  (resident data; excludes H2D/D2H and allocations).
- **step wall / cert wall**: host wall time around the full CUDA call —
  H2D copies + kernels + D2H + cudaMalloc/Free (transfer-inclusive).
- **commit+FS**: CPU SHA-256 leaf hashing + transcript + challenge derivation.
- **verify**: full independent stdlib verification, single process, wall ms.
- Protocol: 2 warmup launches + 5 measured launches per config; median with
  min/max range reported. GPU runs serialized under
  `flock /tmp/requant-new-studies-gpu.lock`, `CUDA_VISIBLE_DEVICES=0`.

## 6. Inputs

Synthetic, seeded: `random.Random(f"{schema}:{seed}:{d}:{r}:{b}")`, uniform
integers in the contract ranges. Two independent cohorts: seeds 20261119 and
987654321. No downloads, no real model weights.