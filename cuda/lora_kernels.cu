// gpu-lora-proof: exact integer LoRA forward+backward step + Freivalds
// certificate vectors on SM120 (Blackwell, RTX 5060 Ti).
//
// Execution paths (per logical GEMM, see contract.py GEMM_PATHS):
//   H, P : INT8 IMMA tensor cores via exact base-128 limb split of W
//          (W = 128*W1 + W0, |W1|<=8, |W0|<=64; int32 accumulation bounds
//          d*127*8 and d*127*64, both << 2^31).  Results recombined in int64.
//   U, Q, dW : INT8 IMMA direct (all operands int8-range; int32 accumulation
//          bounds d*127*64 and b*127*127, both << 2^31).
//   V, R, T8, T9 : exact int64 SIMT GEMM (an operand exceeds int8 range;
//          proven K*max1*max2 bounds < 2^63, no overflow, no float).
// Certificate matvecs: SIMT over F_p, p = 2^61-1, unsigned __int128
// products with Mersenne folding (exact, no division).
//
// Build: nvcc -std=c++17 -O3 -arch=sm_120 -shared -Xcompiler -fPIC \
//        lora_kernels.cu -o liblora.so
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

using namespace nvcuda;

#define P61 ((__uint128_t)(((unsigned long long)1 << 61) - 1))
#define P61LL ((long long)(((unsigned long long)1 << 61) - 1))

// Exact reduction mod 2^61-1 for any 128-bit x: two Mersenne folds plus one
// conditional subtract.  Verified: x < 2^128 -> result < p.
__device__ __forceinline__ unsigned long long fold128(__uint128_t x) {
  x = (x & P61) + (x >> 61);
  x = (x & P61) + (x >> 61);
  if (x >= P61) x -= P61;
  return (unsigned long long)x;
}

// out[i] = sum_j M(i,j) v[j] mod p.  M is a leaf matrix, optionally the
// transpose of the logical operand: trans ? M(i,j)=leaf[j*rows+i]
// : M(i,j)=leaf[i*inner+j].  Leaf entries may be negative but |entry| < p
// (proven), so one conditional add normalizes them into [0,p).
__global__ void mv_kernel(long long *out, const long long *M,
                          const long long *v, int rows, int inner, int trans) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  __uint128_t acc = 0;
  int cnt = 0;
  for (int j = 0; j < inner; j++) {
    long long mij = trans ? M[(long long)j * rows + i]
                          : M[(long long)i * inner + j];
    unsigned long long a = (unsigned long long)(mij >= 0 ? mij : mij + P61LL);
    acc += (__uint128_t)a * (unsigned long long)v[j];
    if (++cnt == 16) { acc = fold128(acc); cnt = 0; }
  }
  out[i] = (long long)fold128(acc);
}

// Exact int64 SIMT GEMM: C(m x n) = A_op(m x k) * B_op(k x n).
// ta/tb: operand is the transpose of the stored leaf.
// Bounds proven < 2^63 per element, so int64 accumulation is exact.
__global__ void gemm_i64(int m, int n, int k, const long long *A,
                         const long long *B, long long *C, int ta, int tb) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m * n) return;
  int i = idx / n, j = idx % n;
  long long acc = 0;
  for (int t = 0; t < k; t++) {
    long long a = ta ? A[(long long)t * m + i] : A[(long long)i * k + t];
    long long b = tb ? B[(long long)j * k + t] : B[(long long)t * n + j];
    acc += a * b;
  }
  C[idx] = acc;
}

// INT8 IMMA GEMM via WMMA 16x16x16 tiles, int32 accumulation.
// m, n, k must be multiples of 16 (zero padding is exact).
// AMAJ/BMAJ are wmma::row_major or wmma::col_major.
template <class AMAJ, class BMAJ>
__global__ void imma_gemm(int m, int n, int k, const int8_t *A,
                          const int8_t *B, int32_t *C, int lda, int ldb) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int tiles_n = n / 16;
  int ti = warp / tiles_n, tj = warp % tiles_n;
  if (ti * 16 >= m) return;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, AMAJ> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, BMAJ> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> fc;
  wmma::fill_fragment(fc, 0);
  for (int t = 0; t < k; t += 16) {
    const int8_t *ap = std::is_same<AMAJ, wmma::row_major>::value
                           ? A + (long long)ti * 16 * lda + t
                           : A + (long long)t * lda + ti * 16;
    const int8_t *bp = std::is_same<BMAJ, wmma::row_major>::value
                           ? B + (long long)t * ldb + tj * 16
                           : B + (long long)tj * 16 * ldb + t;
    wmma::load_matrix_sync(fa, ap, lda);
    wmma::load_matrix_sync(fb, bp, ldb);
    wmma::mma_sync(fc, fa, fb, fc);
  }
  wmma::store_matrix_sync(C + (long long)ti * 16 * n + tj * 16, fc, n,
                          wmma::mem_row_major);
}

// Pack int64 -> int8 with zero padding (values range-checked on host).
__global__ void pack_i8(int8_t *dst, const long long *src, int rows, int cols,
                        int dld, int sld) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  int i = idx / cols, j = idx % cols;
  dst[(long long)i * dld + j] = (int8_t)src[(long long)i * sld + j];
}

// Exact base-128 limb split: W = 128*W1 + W0, W1 = floor((W+64)/128) in
// [-8,8], W0 in [-64,63].  Arithmetic right shift on two's complement.
__global__ void split_w(const long long *W, int8_t *W0, int8_t *W1, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  long long w = W[idx];
  long long w1 = (w + 64) >> 7;
  long long w0 = w - 128 * w1;
  W1[idx] = (int8_t)w1;
  W0[idx] = (int8_t)w0;
}

// out(b x d) = 128*hi + lo from padded int32 IMMA results (ld >= cols).
__global__ void limb_combine(long long *out, const int32_t *hi,
                             const int32_t *lo, int rows, int cols, int ld) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  int i = idx / cols, j = idx % cols;
  out[idx] = 128LL * hi[(long long)i * ld + j] + lo[(long long)i * ld + j];
}

__global__ void cast_i32(long long *out, const int32_t *in, int rows, int cols,
                         int ld) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  int i = idx / cols, j = idx % cols;
  out[idx] = in[(long long)i * ld + j];
}

__global__ void axpy(long long *out, const long long *a, const long long *b,
                     long long s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + s * b[i];
}

__global__ void scale_k(long long *out, const long long *a, long long s,
                        int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = s * a[i];
}

static inline int grid(int n, int tpb) { return (n + tpb - 1) / tpb; }

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),      \
              __FILE__, __LINE__);                                             \
      return (int)e_;                                                          \
    }                                                                          \
  } while (0)

// Leaf indices (must match contract.LEAF_ORDER).
enum { LW = 0, LA, LB, Lx, Lg, LH, LU, LV, Ly, LP, LQ, LR, LT8, LT9, LdX,
       LdW, LdA, LdB, NLEAF };

struct GemmMeta { int m, n, k, a_leaf, b_leaf, c_leaf, ta, tb; };

static void gemm_table(int d, int r, int b, GemmMeta *G) {
  G[0] = {b, d, d, Lx, LW, LH, 0, 0};   // H = x W        (IMMA limb128)
  G[1] = {b, r, d, Lx, LA, LU, 0, 1};   // U = x A^T      (IMMA)
  G[2] = {b, d, r, LU, LB, LV, 0, 1};   // V = U B^T      (SIMT)
  G[3] = {b, d, d, Lg, LW, LP, 0, 1};   // P = g W^T      (IMMA limb128)
  G[4] = {b, r, d, Lg, LB, LQ, 0, 0};   // Q = g B        (IMMA)
  G[5] = {b, d, r, LQ, LA, LR, 0, 0};   // R = Q A        (SIMT)
  G[6] = {d, d, b, Lx, Lg, LdW, 1, 0};  // dW = x^T g     (IMMA)
  G[7] = {r, d, b, LQ, Lx, LT8, 1, 0};  // T8 = Q^T x     (SIMT)
  G[8] = {d, r, b, Lg, LU, LT9, 1, 0};  // T9 = g^T U     (SIMT)
}

extern "C" int lora_info(char *name_buf, int *cc_major, int *cc_minor,
                         int *driver_version, int *runtime_version) {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  snprintf(name_buf, 256, "%s", prop.name);
  *cc_major = prop.major;
  *cc_minor = prop.minor;
  CK(cudaDriverGetVersion(driver_version));
  CK(cudaRuntimeGetVersion(runtime_version));
  return 0;
}

// One exact LoRA forward+backward step.  All host buffers are int64.
// kernel_ms: CUDA-event span of the compute kernels only (excludes H2D/D2H).
// paths[9]: execution path per logical GEMM (0 SIMT, 1 IMMA, 2 IMMA-limb).
extern "C" int lora_step(int d, int r, int b, long long s,
                         const long long *x, const long long *W,
                         const long long *A, const long long *B,
                         const long long *g,
                         long long *H, long long *U, long long *V,
                         long long *y, long long *P, long long *Q,
                         long long *R, long long *T8, long long *T9,
                         long long *dX, long long *dW, long long *dA,
                         long long *dB, double *kernel_ms, int *paths) {
  const int mp = ((b + 15) / 16) * 16;   // padded batch
  const int np = ((r + 15) / 16) * 16;   // padded rank
  const long long dd = (long long)d * d, bd = (long long)b * d;
  const long long br = (long long)b * r, rd = (long long)r * d;
  const long long dr = (long long)d * r;

  long long *dx, *dW_, *dA_, *dB_, *dg, *dH, *dU, *dV, *dy, *dP, *dQ, *dR,
      *dT8, *dT9, *ddX, *ddW, *ddA, *ddB;
  int8_t *x8, *g8, *A8, *B8, *W08, *W18;
  int32_t *H0i, *H1i, *P0i, *P1i, *Ui, *Qi, *dWi;
  cudaEvent_t ev0, ev1;

  CK(cudaMalloc(&dx, bd * 8));      CK(cudaMalloc(&dW_, dd * 8));
  CK(cudaMalloc(&dA_, rd * 8));     CK(cudaMalloc(&dB_, dr * 8));
  CK(cudaMalloc(&dg, bd * 8));
  CK(cudaMalloc(&dH, bd * 8));      CK(cudaMalloc(&dU, br * 8));
  CK(cudaMalloc(&dV, bd * 8));      CK(cudaMalloc(&dy, bd * 8));
  CK(cudaMalloc(&dP, bd * 8));      CK(cudaMalloc(&dQ, br * 8));
  CK(cudaMalloc(&dR, bd * 8));      CK(cudaMalloc(&dT8, rd * 8));
  CK(cudaMalloc(&dT9, dr * 8));     CK(cudaMalloc(&ddX, bd * 8));
  CK(cudaMalloc(&ddW, dd * 8));     CK(cudaMalloc(&ddA, rd * 8));
  CK(cudaMalloc(&ddB, dr * 8));
  CK(cudaMalloc(&x8, (long long)mp * d));   CK(cudaMemset(x8, 0, (long long)mp * d));
  CK(cudaMalloc(&g8, (long long)mp * d));   CK(cudaMemset(g8, 0, (long long)mp * d));
  CK(cudaMalloc(&A8, (long long)np * d));   CK(cudaMemset(A8, 0, (long long)np * d));
  CK(cudaMalloc(&B8, (long long)d * np));   CK(cudaMemset(B8, 0, (long long)d * np));
  CK(cudaMalloc(&W08, dd));         CK(cudaMalloc(&W18, dd));
  CK(cudaMalloc(&H0i, (long long)mp * d * 4));
  CK(cudaMalloc(&H1i, (long long)mp * d * 4));
  CK(cudaMalloc(&P0i, (long long)mp * d * 4));
  CK(cudaMalloc(&P1i, (long long)mp * d * 4));
  CK(cudaMalloc(&Ui, (long long)mp * np * 4));
  CK(cudaMalloc(&Qi, (long long)mp * np * 4));
  CK(cudaMalloc(&dWi, dd * 4));
  CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

  CK(cudaMemcpy(dx, x, bd * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dW_, W, dd * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dA_, A, rd * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB_, B, dr * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dg, g, bd * 8, cudaMemcpyHostToDevice));

  CK(cudaEventRecord(ev0));
  const int T = 256;
  pack_i8<<<grid(b * d, T), T>>>(x8, dx, b, d, d, d);
  pack_i8<<<grid(b * d, T), T>>>(g8, dg, b, d, d, d);
  pack_i8<<<grid(r * d, T), T>>>(A8, dA_, r, d, d, d);
  pack_i8<<<grid(d * r, T), T>>>(B8, dB_, d, r, np, r);
  split_w<<<grid(d * d, T), T>>>(dW_, W08, W18, d * d);

  // INT8 IMMA GEMMs (padded dims are multiples of 16; zero padding exact).
  int warps, blk = 128;
  warps = (mp / 16) * (d / 16);
  imma_gemm<wmma::row_major, wmma::row_major><<<grid(warps * 32, blk), blk>>>(
      mp, d, d, x8, W08, H0i, d, d);
  imma_gemm<wmma::row_major, wmma::row_major><<<grid(warps * 32, blk), blk>>>(
      mp, d, d, x8, W18, H1i, d, d);
  imma_gemm<wmma::row_major, wmma::col_major><<<grid(warps * 32, blk), blk>>>(
      mp, d, d, g8, W08, P0i, d, d);
  imma_gemm<wmma::row_major, wmma::col_major><<<grid(warps * 32, blk), blk>>>(
      mp, d, d, g8, W18, P1i, d, d);
  warps = (mp / 16) * (np / 16);
  imma_gemm<wmma::row_major, wmma::col_major><<<grid(warps * 32, blk), blk>>>(
      mp, np, d, x8, A8, Ui, d, d);
  imma_gemm<wmma::row_major, wmma::row_major><<<grid(warps * 32, blk), blk>>>(
      mp, np, d, g8, B8, Qi, d, np);
  warps = (d / 16) * (d / 16);
  imma_gemm<wmma::col_major, wmma::row_major><<<grid(warps * 32, blk), blk>>>(
      d, d, mp, x8, g8, dWi, d, d);

  limb_combine<<<grid(b * d, T), T>>>(dH, H1i, H0i, b, d, d);
  limb_combine<<<grid(b * d, T), T>>>(dP, P1i, P0i, b, d, d);
  cast_i32<<<grid(b * r, T), T>>>(dU, Ui, b, r, np);
  cast_i32<<<grid(b * r, T), T>>>(dQ, Qi, b, r, np);
  cast_i32<<<grid(d * d, T), T>>>(ddW, dWi, d, d, d);

  // Exact int64 SIMT GEMMs (operands exceed int8 range).
  gemm_i64<<<grid(b * d, T), T>>>(b, d, r, dU, dB_, dV, 0, 1);  // V = U B^T
  gemm_i64<<<grid(b * d, T), T>>>(b, d, r, dQ, dA_, dR, 0, 0);  // R = Q A
  gemm_i64<<<grid(r * d, T), T>>>(r, d, b, dQ, dx, dT8, 1, 0);  // T8 = Q^T x
  gemm_i64<<<grid(d * r, T), T>>>(d, r, b, dg, dU, dT9, 1, 0);  // T9 = g^T U

  axpy<<<grid(b * d, T), T>>>(dy, dH, dV, s, b * d);      // y = H + s V
  axpy<<<grid(b * d, T), T>>>(ddX, dP, dR, s, b * d);    // dX = P + s R
  scale_k<<<grid(r * d, T), T>>>(ddA, dT8, s, r * d);    // dA = s T8
  scale_k<<<grid(d * r, T), T>>>(ddB, dT9, s, d * r);    // dB = s T9
  CK(cudaEventRecord(ev1));
  CK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, ev0, ev1));
  *kernel_ms = (double)ms;

  CK(cudaMemcpy(H, dH, bd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(U, dU, br * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(V, dV, bd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(y, dy, bd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(P, dP, bd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(Q, dQ, br * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(R, dR, bd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(T8, dT8, rd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(T9, dT9, dr * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(dX, ddX, bd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(dW, ddW, dd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(dA, ddA, rd * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(dB, ddB, dr * 8, cudaMemcpyDeviceToHost));
  CK(cudaDeviceSynchronize());

  const int pth[9] = {2, 1, 0, 2, 1, 0, 1, 0, 0};
  for (int i = 0; i < 9; i++) paths[i] = pth[i];

  cudaFree(dx); cudaFree(dW_); cudaFree(dA_); cudaFree(dB_); cudaFree(dg);
  cudaFree(dH); cudaFree(dU); cudaFree(dV); cudaFree(dy); cudaFree(dP);
  cudaFree(dQ); cudaFree(dR); cudaFree(dT8); cudaFree(dT9); cudaFree(ddX);
  cudaFree(ddW); cudaFree(ddA); cudaFree(ddB);
  cudaFree(x8); cudaFree(g8); cudaFree(A8); cudaFree(B8); cudaFree(W08);
  cudaFree(W18);
  cudaFree(H0i); cudaFree(H1i); cudaFree(P0i); cudaFree(P1i); cudaFree(Ui);
  cudaFree(Qi); cudaFree(dWi);
  cudaEventDestroy(ev0); cudaEventDestroy(ev1);
  return 0;
}

// Freivalds certificate vectors for all 9 GEMMs x REPS challenges.
// mats: 18 host int64 leaf buffers in LEAF_ORDER.  chal: flat challenge
// buffer (layout contract.challenge_layout).  resp: flat response buffer
// (layout contract.response_layout).  kernel_ms: event span of matvecs only.
extern "C" int lora_cert(int d, int r, int b, const long long *const *mats,
                         const long long *chal, long long *resp,
                         double *kernel_ms) {
  GemmMeta G[9];
  gemm_table(d, r, b, G);
  const long long sizes[NLEAF] = {
      (long long)d * d, (long long)r * d, (long long)d * r, (long long)b * d,
      (long long)b * d, (long long)b * d, (long long)b * r, (long long)b * d,
      (long long)b * d, (long long)b * d, (long long)b * r, (long long)b * d,
      (long long)r * d, (long long)d * r, (long long)b * d, (long long)d * d,
      (long long)r * d, (long long)d * r};

  long long *dm[NLEAF];
  for (int i = 0; i < NLEAF; i++) {
    CK(cudaMalloc(&dm[i], sizes[i] * 8));
    CK(cudaMemcpy(dm[i], mats[i], sizes[i] * 8, cudaMemcpyHostToDevice));
  }
  // Challenge/response flat buffer sizes (mirror contract.py layouts).
  long long nchal = 0, nresp = 0;
  for (int i = 0; i < 9; i++)
    for (int rep = 0; rep < 2; rep++) {
      nchal += G[i].n;
      nresp += G[i].k + 2 * G[i].m;
    }
  long long *dchal, *dresp;
  CK(cudaMalloc(&dchal, nchal * 8));
  CK(cudaMalloc(&dresp, nresp * 8));
  CK(cudaMemcpy(dchal, chal, nchal * 8, cudaMemcpyHostToDevice));

  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
  CK(cudaEventRecord(ev0));
  const int T = 256;
  long long coff = 0, roff = 0;
  for (int i = 0; i < 9; i++) {
    for (int rep = 0; rep < 2; rep++) {
      const GemmMeta &gm = G[i];
      long long *u = dresp + roff, *v = dresp + roff + gm.k,
                *w = dresp + roff + gm.k + gm.m;
      const long long *rv = dchal + coff;
      // u = B_op r  (rows k, inner n)
      mv_kernel<<<grid(gm.k, T), T>>>(u, dm[gm.b_leaf], rv, gm.k, gm.n, gm.tb);
      // v = A_op u  (rows m, inner k)
      mv_kernel<<<grid(gm.m, T), T>>>(v, dm[gm.a_leaf], u, gm.m, gm.k, gm.ta);
      // w = C r     (rows m, inner n)
      mv_kernel<<<grid(gm.m, T), T>>>(w, dm[gm.c_leaf], rv, gm.m, gm.n, 0);
      coff += gm.n;
      roff += gm.k + 2 * gm.m;
    }
  }
  CK(cudaEventRecord(ev1));
  CK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, ev0, ev1));
  *kernel_ms = (double)ms;
  CK(cudaMemcpy(resp, dresp, nresp * 8, cudaMemcpyDeviceToHost));
  CK(cudaDeviceSynchronize());
  for (int i = 0; i < NLEAF; i++) cudaFree(dm[i]);
  cudaFree(dchal); cudaFree(dresp);
  cudaEventDestroy(ev0); cudaEventDestroy(ev1);
  return 0;
}