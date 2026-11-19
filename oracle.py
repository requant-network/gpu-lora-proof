#!/usr/bin/env python3
"""Independent exact-integer CPU oracle for gpu-lora-proof.

Recomputes every claimed output/gradient leaf from the claimed inputs using
a DIFFERENT algorithm than both the CUDA prover and the verifier: rank-1
column accumulation (C += a_col * b_row outer-product updates) with exact
Python integers, no modular arithmetic, no float.  Proven K*max1*max2 bounds
(< 2^63) guarantee no overflow, so this is an exact deterministic replay.

Input authenticity is NOT checked here (the verifier pins input hashes to
the trusted manifest); this oracle checks the arithmetic of the step.

Exit 0 with mismatch count 0, else exit 1.
"""
import argparse
import json
import sys
import time

import contract as C


def rows_of(flat, shape):
    rows, cols = shape
    return [list(flat[i * cols:(i + 1) * cols]) for i in range(rows)]


def gemm_rank1(Arows, Brows, transA, transB, m, n, k):
    """C = A_op B_op via rank-1 updates: for t, C[i][:] += A(i,t)*B(t,:)."""
    if transA:
        Arows = [list(col) for col in zip(*Arows)]
    if transB:
        # B_op(t, :) = column t of stored leaf B (n x k)
        Bt = [list(col) for col in zip(*Brows)]
    else:
        Bt = Brows
    acc = [[0] * n for _ in range(m)]
    for t in range(k):
        bt = Bt[t]
        for i in range(m):
            a = Arows[i][t]
            if a:
                acc[i] = [c + a * b for c, b in zip(acc[i], bt)]
    return acc


def oracle_replay(proof):
    par = proof["params"]
    d, r, b, s = par["d"], par["r"], par["b"], par["s"]
    sh = C.shapes(d, r, b)
    L = {}
    for n in C.LEAF_ORDER:
        e = proof["leaves"][n]
        L[n] = rows_of(C.b64_decode_i64(e["data_b64"]), tuple(e["shape"]))
    env = {"b": b, "d": d, "r": r}
    computed = {}
    for (out, an, at, bn, bt, ms, ns, ks) in C.GEMM_SPEC:
        m, n, k = env[ms], env[ns], env[ks]
        computed[out] = gemm_rank1(L[an], L[bn], at, bt, m, n, k)
    for (out, a, bb, add) in C.COMBINES:
        if add:
            computed[out] = [[u + s * v for u, v in zip(ra, rb)]
                             for ra, rb in zip(computed[a], computed[bb])]
        else:
            computed[out] = [[s * v for v in ra] for ra in computed[a]]
    mismatches = {}
    total_elements = 0
    for out in computed:
        bad = sum(1 for ra, rb in zip(computed[out], L[out])
                  for u, v in zip(ra, rb) if u != v)
        nel = sh[out][0] * sh[out][1]
        total_elements += nel
        if bad:
            mismatches[out] = bad
    return mismatches, total_elements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proof")
    ap.add_argument("--json", help="write oracle report JSON here")
    a = ap.parse_args()
    with open(a.proof) as f:
        proof = json.load(f)
    t0 = time.perf_counter()
    mismatches, nel = oracle_replay(proof)
    ms = (time.perf_counter() - t0) * 1e3
    report = {"mismatches": mismatches, "elements_checked": nel,
              "oracle_ms": ms, "ok": not mismatches}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    if mismatches:
        print(f"ORACLE FAIL: {mismatches}")
        sys.exit(1)
    print(f"ORACLE OK: {nel} elements exact ({ms:.1f} ms)")
    sys.exit(0)


if __name__ == "__main__":
    main()