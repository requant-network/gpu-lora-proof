"""Shared proof CONTRACT for gpu-lora-proof (serialization + Fiat-Shamir only).

This module defines the byte encoding, transcript, challenge derivation and
GEMM specification table.  It contains NO verification arithmetic: the GPU
prover (CUDA) and the CPU verifier (pure stdlib map/sum modular dot products)
implement the mathematics independently.  Sharing the serialization contract
is required for both sides to agree on the transcript; sharing check code
would defeat independence, so none is shared.

Arithmetic contract (all exact integers):
  x: b x d int8 activations, |x| <= 127
  W: d x d base weights,     |W| <= 1024  (int16-range)
  A: r x d LoRA factor,      |A| <= 64
  B: d x r LoRA factor,      |B| <= 64
  g: b x d loss gradient,    |g| <= 127   (int8-range)
  s: fixed positive integer scale (8 in this study)

  Forward:  H = x W ;  U = x A^T ;  V = U B^T ;  y = H + s V
  Backward: P = g W^T ; Q = g B ; R = Q A ;
            dX = P + s R ; dW = x^T g ;
            T8 = Q^T x ; dA = s T8 ; T9 = g^T U ; dB = s T9

Field: p = 2^61 - 1 (Mersenne prime).  Two independent Fiat-Shamir challenge
vectors per GEMM.  Soundness: every claimed integer matrix has |entry| < p/2
(proven bounds), so a wrong integer claim is nonzero mod p; each Freivalds
check fails to catch it with probability <= 1/p; union bound over 9 GEMMs x
2 reps gives <= 9/p^2 < 2^-118  (target was <= 2^-63).
"""
import base64
import hashlib
import json
import struct
import sys

assert sys.byteorder == "little", "canonical encoding is little-endian int64"

SCHEMA = "gpu-lora-proof/v1"
P = (1 << 61) - 1          # Mersenne prime 2^61-1
REPS = 2                   # independent challenge vectors per GEMM
S_DEFAULT = 8

X_MAX, W_MAX, A_MAX, B_MAX, G_MAX = 127, 1024, 64, 64, 127

# Fixed leaf order for the commitment transcript.
LEAF_ORDER = ["W", "A", "B", "x", "g",
              "H", "U", "V", "y",
              "P", "Q", "R", "T8", "T9", "dX", "dW", "dA", "dB"]
INPUT_LEAVES = ["W", "A", "B", "x", "g"]


def shapes(d, r, b):
    return {"W": (d, d), "A": (r, d), "B": (d, r), "x": (b, d), "g": (b, d),
            "H": (b, d), "U": (b, r), "V": (b, d), "y": (b, d),
            "P": (b, d), "Q": (b, r), "R": (b, d), "T8": (r, d), "T9": (d, r),
            "dX": (b, d), "dW": (d, d), "dA": (r, d), "dB": (d, r)}


# GEMM table: (output, A_leaf, A_transposed, B_leaf, B_transposed, m, n, k)
# dimensions are symbolic: 'b', 'd', 'r'.  C = A_op @ B_op exactly.
GEMM_SPEC = [
    ("H",  "x", False, "W", False, "b", "d", "d"),
    ("U",  "x", False, "A", True,  "b", "r", "d"),
    ("V",  "U", False, "B", True,  "b", "d", "r"),
    ("P",  "g", False, "W", True,  "b", "d", "d"),
    ("Q",  "g", False, "B", False, "b", "r", "d"),
    ("R",  "Q", False, "A", False, "b", "d", "r"),
    ("dW", "x", True,  "g", False, "d", "d", "b"),
    ("T8", "Q", True,  "x", False, "r", "d", "b"),
    ("T9", "g", True,  "U", False, "d", "r", "b"),
]

# Exact elementwise combination checks (not GEMMs): out = a + s*b or s*a.
COMBINES = [("y", "H", "V", True), ("dX", "P", "R", True),
            ("dA", "T8", None, False), ("dB", "T9", None, False)]

# GPU execution path per GEMM (directive: native SM120 tensor cores where
# operands fit int8 with proven int32 accumulation bounds):
#   2 = INT8 IMMA via exact base-128 limb split of W (W = 128*W1 + W0)
#   1 = INT8 IMMA direct (both operands int8-range)
#   0 = exact int64 SIMT (an operand exceeds int8 range)
GEMM_PATHS = [2, 1, 0, 2, 1, 0, 1, 0, 0]
PATH_NAMES = {0: "int64-SIMT", 1: "int8-IMMA", 2: "int8-IMMA-limb128"}


def bounds(d, r, b, s):
    """Worst-case absolute-value bound for every leaf (K*max1*max2 per GEMM)."""
    H = d * X_MAX * W_MAX
    U = d * X_MAX * A_MAX
    V = r * U * B_MAX
    P = d * G_MAX * W_MAX
    Q = d * G_MAX * B_MAX
    R = r * Q * A_MAX
    dW = b * X_MAX * G_MAX
    T8 = b * Q * X_MAX
    T9 = b * G_MAX * U
    return {"W": W_MAX, "A": A_MAX, "B": B_MAX, "x": X_MAX, "g": G_MAX,
            "H": H, "U": U, "V": V, "y": H + s * V,
            "P": P, "Q": Q, "R": R, "T8": T8, "T9": T9,
            "dX": P + s * R, "dW": dW, "dA": s * T8, "dB": s * T9}


def imma_int32_bounds(d, b):
    """Accumulation bounds for the INT8 IMMA (int32 accumulator) GEMMs."""
    return {"xW1": d * X_MAX * 8, "xW0": d * X_MAX * 64,
            "gW1": d * G_MAX * 8, "gW0": d * G_MAX * 64,
            "U": d * X_MAX * A_MAX, "Q": d * G_MAX * B_MAX,
            "dW": b * X_MAX * G_MAX}


def assert_contract(d, r, b, s):
    bd = bounds(d, r, b, s)
    for name, v in bd.items():
        assert v < (1 << 63), f"int64 overflow risk: {name}={v}"
        assert v < P // 2, f"field embedding not faithful: {name}={v}"
    for name, v in imma_int32_bounds(d, b).items():
        assert v < (1 << 31), f"int32 IMMA overflow risk: {name}={v}"
    assert d % 16 == 0, "d must be a multiple of 16 for WMMA tiles"


def params_dict(d, r, b, s, seed):
    return {"schema": SCHEMA, "d": d, "r": r, "b": b, "s": s, "seed": seed,
            "x_max": X_MAX, "w_max": W_MAX, "a_max": A_MAX, "b_max": B_MAX,
            "g_max": G_MAX, "field_prime": "2^61-1", "reps": REPS}


def step_id(d, r, b, s, seed):
    return f"lora-step-d{d}-r{r}-b{b}-s{s}-seed{seed}"


def params_canonical(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":")).encode("ascii")


def leaf_bytes(name, shape, data_bytes):
    return (name.encode("ascii") + b"\x00" + struct.pack("<I", len(shape))
            + struct.pack("<" + "I" * len(shape), *shape) + data_bytes)


def leaf_hash(name, shape, data_bytes):
    return hashlib.sha256(leaf_bytes(name, shape, data_bytes)).digest()


def transcript(step_id_str, params, leaf_hashes_raw):
    """leaf_hashes_raw: dict name -> 32 raw bytes, consumed in LEAF_ORDER."""
    return (SCHEMA.encode("ascii") + b"\x00" + step_id_str.encode("ascii")
            + b"\x00" + params_canonical(params) + b"\x00"
            + b"".join(leaf_hashes_raw[n] for n in LEAF_ORDER))


def challenge_vector(trans, gemm_idx, rep, n):
    """Fiat-Shamir challenge r in F_p^n, SHA-256 counter mode (4 u64/block)."""
    out = []
    block = 0
    while len(out) < n:
        dig = hashlib.sha256(trans + struct.pack("<III", gemm_idx, rep, block)).digest()
        for v in struct.unpack("<4Q", dig):
            out.append(v % P)
            if len(out) == n:
                break
        block += 1
    return out


def gemm_dims(d, r, b):
    env = {"b": b, "d": d, "r": r}
    return [(env[m], env[n], env[k]) for (_, _, _, _, _, m, n, k) in GEMM_SPEC]


def challenge_layout(d, r, b):
    """[(gemm_idx, rep, offset, n)] into the flat int64 challenge buffer."""
    lay, off = [], 0
    for i, (m, n, k) in enumerate(gemm_dims(d, r, b)):
        for rep in range(REPS):
            lay.append((i, rep, off, n))
            off += n
    return lay, off


def response_layout(d, r, b):
    """[(gemm_idx, rep, u_off, v_off, w_off, k, m)] flat int64 buffer.

    u = B_op r (length k), v = A_op u (length m), w = C r (length m).
    """
    lay, off = [], 0
    for i, (m, n, k) in enumerate(gemm_dims(d, r, b)):
        for rep in range(REPS):
            lay.append((i, rep, off, off + k, off + k + m, k, m))
            off += k + 2 * m
    return lay, off


def b64_encode_i64(flat):
    return base64.b64encode(struct.pack("<%dq" % len(flat), *flat)).decode("ascii")


def b64_decode_i64(s):
    raw = base64.b64decode(s)
    if len(raw) % 8:
        raise ValueError("int64 buffer length not a multiple of 8")
    return list(struct.unpack("<%dq" % (len(raw) // 8), raw))