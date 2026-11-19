# RESULTS — gpu-lora-proof

All numbers below were measured on the server described here; none are extrapolated. Generated from `results/results.json` by `experiments/make_report.py`.

## Environment

- GPU: **NVIDIA GeForce RTX 5060 Ti** (compute capability **12.0**, SM120 Blackwell), GPU 0 only (`CUDA_VISIBLE_DEVICES=0`)
- Driver: 610.43.03; CUDA driver/runtime API version: 13030
- Compiler: Build cuda_13.3.r13.3/compiler.38244171_0 (`-std=c++17 -O3 -arch=sm_120`)
- Host: me4me, Python 3.12.3 (stdlib only)
- Run (UTC): 2026-11-19T16:06:28Z
- Compiled library SHA-256: `63bc6730e35621b96f1b5723c7e580914ce838349d32a2304ec4e7723f0af4b2`

## Summary counts

- Configurations: 24 (d in {256,512,1024} x r in {8,16} x batch in {1,16} x 2 seeds {20261119, 987654321})
- Honest proofs accepted by independent CPU verifier (YES, exit 0): **24/24**
- Exact CPU oracle replays with zero mismatches: **24/24** (12,434,208 elements checked exactly)
- Negative controls rejected: **276/276** (11 single-proof tampers per config + 12 cross-seed stale replays)
- Total exact verifier checks on honest proofs: 3,312 (138 per proof)
- Soundness bound: 9/p^2 with p=2^61-1 => < 2^-118 (target <= 2^-63)

## Timing protocol

2 warmup launches + 5 measured launches per configuration; table shows median with min/max range (ms). Boundaries: `step wall` / `cert wall` are transfer-inclusive host wall time around the CUDA call (H2D + kernels + D2H + alloc/free); `step kernel` / `cert kernel` are CUDA-event spans of compute kernels only; `commit+FS` is CPU SHA-256 commitment and Fiat-Shamir challenge derivation; `verify` is the full independent pure-stdlib CPU verification (hashes + 18 Freivalds checks + 54 response bindings + 4 exact combines + pins).

## Per-configuration measurements (median [min, max], ms)

| config (seed) | proof MB | step wall | step kernel | cert wall | cert kernel | commit+FS | CPU verify |
|---|---|---|---|---|---|---|---|
| d256 r8 b1 (20261119) | 1.60 | 5.84 [5.47, 6.32] | 0.070 [0.064, 0.075] | 4.02 [3.75, 4.63] | 0.626 [0.624, 0.640] | 4.11 [4.05, 4.24] | 60.5 |
| d256 r8 b1 (987654321) | 1.60 | 6.75 [6.13, 7.31] | 0.072 [0.068, 0.082] | 4.46 [3.82, 4.88] | 0.640 [0.626, 0.646] | 4.68 [4.29, 5.12] | 57.6 |
| d256 r8 b16 (20261119) | 1.93 | 7.77 [7.07, 8.31] | 0.075 [0.068, 0.082] | 5.23 [5.15, 5.47] | 0.661 [0.660, 0.664] | 5.59 [5.44, 5.76] | 69.0 |
| d256 r8 b16 (987654321) | 1.93 | 8.19 [7.57, 9.54] | 0.071 [0.063, 0.086] | 5.25 [4.64, 5.98] | 0.661 [0.649, 0.664] | 5.39 [5.23, 6.45] | 75.0 |
| d256 r16 b1 (20261119) | 1.73 | 7.74 [6.90, 8.78] | 0.079 [0.069, 0.091] | 5.21 [4.48, 5.68] | 0.655 [0.654, 0.659] | 5.49 [5.06, 5.82] | 64.5 |
| d256 r16 b1 (987654321) | 1.73 | 7.59 [6.45, 8.16] | 0.083 [0.068, 0.087] | 5.31 [4.51, 5.77] | 0.660 [0.655, 0.662] | 5.63 [4.79, 5.79] | 68.3 |
| d256 r16 b16 (20261119) | 2.07 | 8.24 [7.55, 8.87] | 0.072 [0.061, 0.086] | 5.39 [4.64, 6.00] | 0.677 [0.664, 0.679] | 5.84 [5.22, 6.30] | 73.5 |
| d256 r16 b16 (987654321) | 2.07 | 9.40 [8.39, 10.15] | 0.071 [0.062, 0.080] | 5.63 [4.88, 7.50] | 0.675 [0.664, 0.679] | 6.34 [5.88, 7.60] | 76.9 |
| d512 r8 b1 (20261119) | 5.99 | 28.21 [26.71, 33.34] | 0.079 [0.078, 0.083] | 14.11 [14.03, 14.39] | 1.203 [1.197, 1.204] | 15.94 [15.68, 17.51] | 233.8 |
| d512 r8 b1 (987654321) | 5.99 | 29.34 [29.25, 30.67] | 0.079 [0.078, 0.091] | 17.82 [15.43, 18.42] | 1.203 [1.202, 1.204] | 19.40 [17.09, 20.41] | 233.1 |
| d512 r8 b16 (20261119) | 6.65 | 31.31 [30.86, 40.11] | 0.072 [0.072, 0.073] | 18.94 [15.75, 19.13] | 1.236 [1.236, 1.237] | 20.81 [17.59, 20.83] | 255.1 |
| d512 r8 b16 (987654321) | 6.65 | 32.65 [32.32, 40.23] | 0.072 [0.071, 0.073] | 18.99 [17.32, 19.10] | 1.231 [1.228, 1.233] | 20.68 [19.04, 23.64] | 250.2 |
| d512 r16 b1 (20261119) | 6.25 | 34.50 [30.45, 36.62] | 0.082 [0.080, 0.104] | 18.47 [16.79, 21.52] | 1.218 [1.209, 1.220] | 19.96 [18.62, 20.44] | 230.6 |
| d512 r16 b1 (987654321) | 6.25 | 30.34 [29.94, 38.77] | 0.081 [0.079, 0.082] | 18.12 [15.91, 18.82] | 1.212 [1.210, 1.220] | 19.64 [17.57, 20.57] | 231.8 |
| d512 r16 b16 (20261119) | 6.92 | 32.24 [32.16, 34.53] | 0.074 [0.073, 0.075] | 19.84 [16.58, 19.93] | 1.252 [1.251, 1.254] | 21.61 [18.42, 21.75] | 259.1 |
| d512 r16 b16 (987654321) | 6.92 | 33.21 [32.86, 34.72] | 0.073 [0.071, 0.082] | 18.73 [17.00, 20.13] | 1.252 [1.241, 1.252] | 20.78 [18.81, 22.02] | 257.0 |
| d1024 r8 b1 (20261119) | 23.16 | 116.03 [112.17, 127.84] | 0.123 [0.121, 0.125] | 48.63 [48.32, 48.85] | 2.300 [2.299, 2.341] | 57.39 [57.19, 58.61] | 939.8 |
| d1024 r8 b1 (987654321) | 23.16 | 127.37 [116.30, 147.75] | 0.124 [0.122, 0.125] | 64.84 [53.08, 64.89] | 2.310 [2.307, 2.320] | 73.47 [61.56, 74.29] | 926.2 |
| d1024 r8 b16 (20261119) | 24.48 | 134.40 [123.60, 162.45] | 0.110 [0.109, 0.110] | 67.63 [55.20, 68.11] | 2.372 [2.369, 2.423] | 76.94 [64.04, 77.10] | 982.0 |
| d1024 r8 b16 (987654321) | 24.48 | 138.23 [126.45, 145.47] | 0.110 [0.109, 0.110] | 68.66 [55.80, 69.39] | 2.371 [2.370, 2.431] | 77.67 [64.69, 93.37] | 1004.9 |
| d1024 r16 b1 (20261119) | 23.69 | 123.79 [120.17, 140.15] | 0.126 [0.125, 0.127] | 65.81 [56.91, 66.15] | 2.343 [2.341, 2.354] | 74.28 [65.58, 75.36] | 941.4 |
| d1024 r16 b1 (987654321) | 23.69 | 115.68 [114.88, 147.26] | 0.127 [0.125, 0.127] | 63.45 [55.02, 64.14] | 2.343 [2.342, 2.344] | 72.46 [63.81, 73.48] | 957.8 |
| d1024 r16 b16 (20261119) | 25.01 | 123.50 [121.87, 132.53] | 0.115 [0.114, 0.115] | 66.11 [56.67, 66.37] | 2.404 [2.402, 2.415] | 75.58 [65.76, 76.16] | 1020.8 |
| d1024 r16 b16 (987654321) | 25.01 | 121.69 [118.57, 122.60] | 0.115 [0.114, 0.117] | 68.97 [58.26, 69.43] | 2.403 [2.402, 2.406] | 78.08 [67.54, 80.65] | 1014.8 |

## GPU execution path per logical GEMM

Paths recorded by the prover for every run: `[2,1,0,2,1,0,1,0,0]` (0 = exact int64 SIMT, 1 = INT8 IMMA tensor core, 2 = INT8 IMMA via exact base-128 limb split of W).

| GEMM | expression | path | why |
|---|---|---|---|
| H | x W | IMMA-limb128 | W exceeds int8; W=128*W1+W0, both limbs int8; int32 acc bounds d*127*8, d*127*64 |
| U | x A^T | IMMA | x,A int8-range; int32 acc bound d*127*64 = 8,323,072 << 2^31 |
| V | U B^T | int64 SIMT | U up to 8.3e6 exceeds int8; int64 bound r*U*64 < 2^34 |
| P | g W^T | IMMA-limb128 | same limb split of W; int32 acc bounds as H |
| Q | g B | IMMA | g,B int8-range; int32 acc bound d*127*64 |
| R | Q A | int64 SIMT | Q up to 8.3e6 exceeds int8 |
| dW | x^T g | IMMA | x,g int8; int32 acc bound b*127*127 = 258,064 |
| T8 | Q^T x | int64 SIMT | Q exceeds int8; int64 bound b*Q*127 < 2^34 |
| T9 | g^T U | int64 SIMT | U exceeds int8; int64 bound b*127*U < 2^34 |

Certificate matvecs (54 per proof) run as SIMT over F_p (p=2^61-1) with unsigned __int128 products and Mersenne folding — field arithmetic, not tensor-core territory.

## Negative controls (all rejected)

| control | runs | rejected | typical rejection reason |
|---|---|---|---|
| flip_dX_coherent | 24 | 24 | exact combine failed: dX |
| alter_W_coherent | 24 | 24 | input pin mismatch W |
| wrong_y_coherent | 24 | 24 | exact combine failed: y |
| wrong_dW_coherent | 24 | 24 | Freivalds identity failed GEMM 6 (dW) rep 0 |
| flip_U_coherent | 24 | 24 | Freivalds identity failed GEMM 1 (U) rep 0 |
| truncated_responses | 24 | 24 | undecodable responses: 'responses_b64' |
| truncated_leaf | 24 | 24 | undecodable leaf W: Incorrect padding |
| wrong_step_id | 24 | 24 | step_id mismatch |
| wrong_scale | 24 | 24 | params mismatch |
| tamper_response_v | 24 | 24 | response v binding failed GEMM 0 rep 0 |
| wrong_x_coherent | 24 | 24 | input pin mismatch x |
| stale_replay_cross_seed | 12 | 12 | step_id mismatch |

Coherent tampers (flip_dX, alter_W, wrong_y, wrong_dW, flip_U, wrong_x) recompute every SHA-256 commitment, rederive all Fiat-Shamir challenges and recompute all 54 response vectors for the tampered claims — a maximally capable re-sealing attacker. Rejection therefore comes from the arithmetic checks (Freivalds / exact combines) or the trusted input pins, not from stale hashes.

## Verifier check census (per honest proof)

| check class | count |
|---|---|
| schema | 1 |
| step_id | 1 |
| params | 1 |
| shapes | 18 |
| leaf_hashes | 18 |
| input_pins | 5 |
| range_bounds | 18 |
| freivalds | 18 |
| response_bindings | 54 |
| combines | 4 |
| **total** | **138** |

Exact elementwise combine checks additionally cover 4,608 elements per proof at d=256 (scales with b*d + r*d + d*r).

## Proof size

Proof = 18 committed int64 leaves (base64) + 54 challenge response vectors + metadata. Dominated by W and dW (d^2 int64 each): 1.60 MB at d=256,b=1 up to 25.01 MB at d=1024,b=16. This is a transparent (non-succinct) proof; no claim of zk-SNARK-sized certificates is made.

## Source SHA-256 (executed files)

- `contract.py`: `3b8af2b46c6eac3d4e53b7ae916b3c3d26a888f0afbd74bb960c77b28458aec5`
- `prover.py`: `1fc457888d2a580d498bf1e768b9af18421a0a224b44278c92b4d1a450c8873e`
- `verifier.py`: `72fc81f77742bcd1e1f9c53177ee2c6108159e8f32367e8452376a59b2789559`
- `oracle.py`: `2579cb30f8555d4ca5348c9caeec06c37eababbc3e262fe3f08dd98aeef2ccb7`
- `tamper.py`: `faa6f0cf8312dc386a0f43eeea0d23349ce56ffae77b880d4a40ba0531ec586f`
- `run.py`: `dbcff3a79e032da5a840ab22cf65dbf5c533b3b15ba2430848a1cab217a11237`
- `cuda/lora_kernels.cu`: `21f95df32493924122faab228cb282619e9d4fd3c9088156751608cc9affc3f7`
