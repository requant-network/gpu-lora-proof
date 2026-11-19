#!/usr/bin/env python3
"""Prover for gpu-lora-proof: exact integer LoRA step + Freivalds certificate.

Backends:
  gpu  -- CUDA library (SM120 INT8 IMMA + int64 SIMT paths, see cuda/).
  cpu  -- pure-stdlib exact integer reference (used by tests; no GPU).

The prover computes the step, commits every leaf with SHA-256, derives
Fiat-Shamir challenges from the commitment transcript, computes the
certificate vectors (on GPU for the gpu backend), and writes the proof JSON
plus the trusted expected-inputs manifest.  All arithmetic is exact integer;
no floating point is used for any claimed value.
"""
import argparse
import array
import ctypes
import json
import random
import statistics
import sys
import time
from operator import mul

import contract as C


def gen_inputs(d, r, b, seed):
    rng = random.Random(f"{C.SCHEMA}:{seed}:{d}:{r}:{b}")
    def mat(rows, cols, lim):
        return [rng.randint(-lim, lim) for _ in range(rows * cols)]
    return {"W": mat(d, d, C.W_MAX), "A": mat(r, d, C.A_MAX),
            "B": mat(d, r, C.B_MAX), "x": mat(b, d, C.X_MAX),
            "g": mat(b, d, C.G_MAX)}


def rows_of(flat, shape):
    rows, cols = shape
    return [tuple(flat[i * cols:(i + 1) * cols]) for i in range(rows)]


def gemm_exact(Arows, Brows, transA, transB):
    """Exact integer GEMM over Python ints (proven bounds, no overflow)."""
    if transA:
        Arows = list(zip(*Arows))
    Bcols = list(zip(*Brows)) if not transB else list(Brows)
    return [tuple(sum(map(mul, a, bc)) for bc in Bcols) for a in Arows]


def cpu_step(d, r, b, s, inp):
    sh = C.shapes(d, r, b)
    L = {n: rows_of(inp[n], sh[n]) for n in C.INPUT_LEAVES}
    for (out, an, at, bn, bt, *_dims) in C.GEMM_SPEC:
        L[out] = gemm_exact(L[an], L[bn], at, bt)
    for (out, a, bb, add) in C.COMBINES:
        if add:
            L[out] = tuple(tuple(u + s * v for u, v in zip(ra, rb))
                           for ra, rb in zip(L[a], L[bb]))
        else:
            L[out] = tuple(tuple(s * v for v in ra) for ra in L[a])
    return {n: [v for row in L[n] for v in row] for n in C.LEAF_ORDER}


def load_lib(path):
    lib = ctypes.CDLL(path)
    LL = ctypes.POINTER(ctypes.c_longlong)
    lib.lora_info.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int)]
    lib.lora_info.restype = ctypes.c_int
    lib.lora_step.argtypes = ([ctypes.c_int] * 3 + [ctypes.c_longlong]
                              + [LL] * 5 + [LL] * 13
                              + [ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_int)])
    lib.lora_step.restype = ctypes.c_int
    lib.lora_cert.argtypes = ([ctypes.c_int] * 3
                              + [ctypes.POINTER(LL), LL, LL,
                                 ctypes.POINTER(ctypes.c_double)])
    lib.lora_cert.restype = ctypes.c_int
    return lib


def gpu_info(lib):
    name = ctypes.create_string_buffer(256)
    mj, mn, dv, rv = (ctypes.c_int() for _ in range(4))
    rc = lib.lora_info(name, ctypes.byref(mj), ctypes.byref(mn),
                       ctypes.byref(dv), ctypes.byref(rv))
    if rc:
        raise RuntimeError(f"lora_info failed: {rc}")
    return {"gpu_name": name.value.decode(), "compute_cap": f"{mj.value}.{mn.value}",
            "cuda_driver_version": dv.value, "cuda_runtime_version": rv.value}


def _ptr(flat):
    arr = array.array("q", flat)
    ca = (ctypes.c_longlong * len(arr)).from_buffer(arr)
    return arr, ca


def gpu_step(lib, d, r, b, s, inp):
    sh = C.shapes(d, r, b)
    for n, lim in (("x", C.X_MAX), ("g", C.G_MAX), ("W", C.W_MAX),
                   ("A", C.A_MAX), ("B", C.B_MAX)):
        assert max(map(abs, inp[n])) <= lim, f"input {n} out of contract range"
    ins = [_ptr(inp[n]) for n in ("x", "W", "A", "B", "g")]
    outs = [_ptr([0] * (sh[n][0] * sh[n][1]))
            for n in ("H", "U", "V", "y", "P", "Q", "R", "T8", "T9",
                      "dX", "dW", "dA", "dB")]
    kms = ctypes.c_double()
    paths = (ctypes.c_int * 9)()
    rc = lib.lora_step(d, r, b, s, *[ca for _, ca in ins],
                       *[ca for _, ca in outs], ctypes.byref(kms), paths)
    if rc:
        raise RuntimeError(f"lora_step failed: {rc}")
    out = {n: list(a) for (a, _), n in zip(outs, ("H", "U", "V", "y", "P", "Q",
                                                 "R", "T8", "T9", "dX", "dW",
                                                 "dA", "dB"))}
    out.update({n: inp[n] for n in C.INPUT_LEAVES})
    return out, kms.value, list(paths)


def commitments(leaves, sh):
    hashes = {}
    for n in C.LEAF_ORDER:
        data = array.array("q", leaves[n]).tobytes()
        hashes[n] = C.leaf_hash(n, sh[n], data)
    return hashes


def all_challenges(trans, d, r, b):
    lay, total = C.challenge_layout(d, r, b)
    flat = []
    for (i, rep, off, n) in lay:
        flat.extend(C.challenge_vector(trans, i, rep, n))
    assert len(flat) == total
    return flat


def cpu_cert(leaves, sh, chal, d, r, b):
    """Independent-style modular matvecs (stdlib) for the cpu backend."""
    red = {n: [[v % C.P for v in row] for row in rows_of(leaves[n], sh[n])]
           for n in C.LEAF_ORDER}
    tpose = {n: list(zip(*red[n])) for n in C.LEAF_ORDER}
    def mv(leaf, trans, vec):
        rows = tpose[leaf] if trans else red[leaf]
        return [sum(map(mul, row, vec)) % C.P for row in rows]
    resp = []
    for (i, rep, off, n) in C.challenge_layout(d, r, b)[0]:
        out, an, at, bn, bt, *_ = C.GEMM_SPEC[i]
        rv = chal[off:off + n]
        u = mv(bn, bt, rv)
        v = mv(an, at, u)
        w = mv(out, False, rv)
        resp.extend(u); resp.extend(v); resp.extend(w)
    return resp


def gpu_cert(lib, leaves, sh, chal, d, r, b):
    arrs, cas = [], []
    for n in C.LEAF_ORDER:
        a, ca = _ptr(leaves[n])
        arrs.append(a); cas.append(ca)
    LL = ctypes.POINTER(ctypes.c_longlong)
    matarr = (LL * len(C.LEAF_ORDER))()
    for i, ca in enumerate(cas):
        matarr[i] = ctypes.cast(ca, LL)
    _, cca = _ptr(chal)
    nresp = C.response_layout(d, r, b)[1]
    rarr = array.array("q", [0] * nresp)
    rca = (ctypes.c_longlong * nresp).from_buffer(rarr)
    kms = ctypes.c_double()
    rc = lib.lora_cert(d, r, b, matarr, cca, rca, ctypes.byref(kms))
    if rc:
        raise RuntimeError(f"lora_cert failed: {rc}")
    return list(rarr), kms.value


def build_proof(d, r, b, s, seed, backend, lib=None, reps=5, warmup=2):
    C.assert_contract(d, r, b, s)
    sh = C.shapes(d, r, b)
    inp = gen_inputs(d, r, b, seed)
    sid = C.step_id(d, r, b, s, seed)
    params = C.params_dict(d, r, b, s, seed)
    timings = []
    leaves = paths = None
    for it in range(warmup + reps):
        t0 = time.perf_counter()
        if backend == "gpu":
            leaves, step_kms, paths = gpu_step(lib, d, r, b, s, inp)
        else:
            leaves, step_kms, paths = cpu_step(d, r, b, s, inp), None, list(C.GEMM_PATHS)
        t1 = time.perf_counter()
        hashes = commitments(leaves, sh)
        trans = C.transcript(sid, params, hashes)
        chal = all_challenges(trans, d, r, b)
        t2 = time.perf_counter()
        if backend == "gpu":
            resp, cert_kms = gpu_cert(lib, leaves, sh, chal, d, r, b)
        else:
            resp, cert_kms = cpu_cert(leaves, sh, chal, d, r, b), None
        t3 = time.perf_counter()
        if it >= warmup:
            timings.append({"step_wall_ms": (t1 - t0) * 1e3,
                            "step_kernel_ms": step_kms,
                            "commit_fs_ms": (t2 - t1) * 1e3,
                            "cert_wall_ms": (t3 - t2) * 1e3,
                            "cert_kernel_ms": cert_kms,
                            "prover_wall_ms": (t3 - t0) * 1e3})
    def stat(key):
        vals = [t[key] for t in timings if t[key] is not None]
        if not vals:
            return None
        return {"median": statistics.median(vals), "min": min(vals),
                "max": max(vals), "n": len(vals)}
    proof = {
        "schema": C.SCHEMA, "step_id": sid, "params": params,
        "leaf_hashes": {n: hashes[n].hex() for n in C.LEAF_ORDER},
        "leaves": {n: {"shape": list(sh[n]),
                       "data_b64": C.b64_encode_i64(leaves[n])}
                   for n in C.LEAF_ORDER},
        "responses_b64": C.b64_encode_i64(resp),
        "gemm_paths": paths,
        "prover_timing": {k: stat(k) for k in timings[0]},
        "timing_boundary": ("step_wall/cert_wall/prover_wall are transfer-"
                            "inclusive host wall time around the CUDA call; "
                            "step_kernel/cert_kernel are CUDA-event spans of "
                            "compute kernels only (exclude H2D/D2H); "
                            "commit_fs is CPU SHA-256+challenge derivation"),
    }
    expected = {"step_id": sid, "params": params,
                "input_hashes": {n: hashes[n].hex() for n in C.INPUT_LEAVES}}
    return proof, expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--s", type=int, default=C.S_DEFAULT)
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--lib", default="./liblora.so")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()
    lib = load_lib(a.lib) if a.backend == "gpu" else None
    proof, expected = build_proof(a.d, a.r, a.b, a.s, a.seed, a.backend, lib,
                                  a.reps, a.warmup)
    with open(a.out, "w") as f:
        json.dump(proof, f)
    with open(a.expected, "w") as f:
        json.dump(expected, f)
    print(f"proof written: {a.out} ({len(json.dumps(proof))} bytes json)")


if __name__ == "__main__":
    main()