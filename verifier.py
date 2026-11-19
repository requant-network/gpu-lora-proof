#!/usr/bin/env python3
"""Independent CPU verifier for gpu-lora-proof (pure Python stdlib).

DIFFERENT code path from the prover: no CUDA, no prover imports.  Arithmetic
is exact modular dot products via sum(map(operator.mul, ...)) over F_p with
p = 2^61-1, plus exact integer elementwise combination checks.  Only the
serialization/Fiat-Shamir contract (contract.py) is shared, as both sides
must agree on transcript bytes.

Checks performed (all counted):
  * schema / step_id / params equal the trusted expected manifest
  * all 18 leaves present with correct shapes and decodable int64 data
  * declared SHA-256 leaf commitments == recomputed hashes
  * input leaves (W, A, B, x, g) pinned to expected input hashes
  * every value within the documented worst-case bound (=> |v| < p/2)
  * Fiat-Shamir challenges recomputed from the commitment transcript
  * per GEMM per rep: u=B r, v=A u, w=C r recomputed independently;
    Freivalds identity v == w; proof response vectors bound to recomputation
  * exact integer combines: y = H + sV, dX = P + sR, dA = s T8, dB = s T9

Prints YES (exit 0) on accept, NO: <reason> (exit 1) on reject.
"""
import argparse
import array
import hashlib
import json
import sys
import time
from operator import mul

import contract as C


class Reject(Exception):
    pass


def _rows(flat, shape):
    rows, cols = shape
    if len(flat) != rows * cols:
        raise Reject(f"leaf data length {len(flat)} != {rows}*{cols}")
    return [tuple(flat[i * cols:(i + 1) * cols]) for i in range(rows)]


def verify(proof, expected):
    t0 = time.perf_counter()
    counts = {"schema": 0, "step_id": 0, "params": 0, "shapes": 0,
              "leaf_hashes": 0, "input_pins": 0, "range_bounds": 0,
              "freivalds": 0, "response_bindings": 0, "combines": 0}
    combine_elements = 0

    def ck(cond, reason, key):
        if not cond:
            raise Reject(reason)
        counts[key] += 1

    ck(proof.get("schema") == C.SCHEMA, "schema mismatch", "schema")
    ck(proof.get("step_id") == expected["step_id"], "step_id mismatch", "step_id")
    ck(proof.get("params") == expected["params"], "params mismatch", "params")
    par = proof["params"]
    d, r, b, s = par["d"], par["r"], par["b"], par["s"]
    C.assert_contract(d, r, b, s)
    sh = C.shapes(d, r, b)
    bd = C.bounds(d, r, b, s)

    leaves, rows = {}, {}
    for n in C.LEAF_ORDER:
        if n not in proof.get("leaves", {}):
            raise Reject(f"missing leaf {n}")
        entry = proof["leaves"][n]
        ck(tuple(entry.get("shape", [])) == sh[n], f"shape mismatch {n}", "shapes")
        try:
            flat = C.b64_decode_i64(entry["data_b64"])
        except Exception as e:
            raise Reject(f"undecodable leaf {n}: {e}")
        leaves[n] = flat
        rows[n] = _rows(flat, sh[n])

    hashes = {}
    for n in C.LEAF_ORDER:
        data = C.leaf_bytes(n, sh[n], array.array("q", leaves[n]).tobytes())
        hashes[n] = hashlib.sha256(data).digest()
        ck(proof.get("leaf_hashes", {}).get(n) == hashes[n].hex(),
           f"commitment hash mismatch {n}", "leaf_hashes")
    for n in C.INPUT_LEAVES:
        ck(hashes[n].hex() == expected["input_hashes"][n],
           f"input pin mismatch {n}", "input_pins")
    for n in C.LEAF_ORDER:
        m = max(map(abs, leaves[n]))
        ck(m <= bd[n], f"range bound violated {n}: {m} > {bd[n]}", "range_bounds")

    trans = C.transcript(proof["step_id"], par, hashes)
    try:
        resp = C.b64_decode_i64(proof["responses_b64"])
    except Exception as e:
        raise Reject(f"undecodable responses: {e}")
    rlay, rtotal = C.response_layout(d, r, b)
    if len(resp) != rtotal:
        raise Reject(f"response buffer length {len(resp)} != {rtotal}")

    # Reduced leaves and transpose views for modular arithmetic.
    red = {n: tuple(tuple(v % C.P for v in row) for row in rows[n])
           for n in C.LEAF_ORDER}
    tpose = {n: tuple(zip(*red[n])) for n in C.LEAF_ORDER}

    def mv(leaf, trans_flag, vec):
        rws = tpose[leaf] if trans_flag else red[leaf]
        return [sum(map(mul, row, vec)) % C.P for row in rws]

    for (i, rep, uo, vo, wo, k, m) in rlay:
        out, an, at, bn, bt, *_ = C.GEMM_SPEC[i]
        n = C.gemm_dims(d, r, b)[i][1]
        rv = C.challenge_vector(trans, i, rep, n)
        u2 = mv(bn, bt, rv)
        v2 = mv(an, at, u2)
        w2 = mv(out, False, rv)
        ck(v2 == w2, f"Freivalds identity failed GEMM {i} ({out}) rep {rep}",
           "freivalds")
        ck(resp[uo:uo + k] == u2, f"response u binding failed GEMM {i} rep {rep}",
           "response_bindings")
        ck(resp[vo:vo + m] == v2, f"response v binding failed GEMM {i} rep {rep}",
           "response_bindings")
        ck(resp[wo:wo + m] == w2, f"response w binding failed GEMM {i} rep {rep}",
           "response_bindings")

    for (out, a, bb, add) in C.COMBINES:
        if add:
            ok = all(u + s * v == w
                     for ra, rb, rw in zip(rows[a], rows[bb], rows[out])
                     for u, v, w in zip(ra, rb, rw))
            nel = sh[out][0] * sh[out][1]
        else:
            ok = all(s * v == w for ra, rw in zip(rows[a], rows[out])
                     for v, w in zip(ra, rw))
            nel = sh[out][0] * sh[out][1]
        ck(ok, f"exact combine failed: {out}", "combines")
        combine_elements += nel

    ms = (time.perf_counter() - t0) * 1e3
    stats = {"checks": counts, "total_checks": sum(counts.values()),
             "combine_elements_checked": combine_elements,
             "freivalds_gemm_checks": counts["freivalds"],
             "verify_ms": ms}
    return True, "all checks passed", stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proof")
    ap.add_argument("expected")
    ap.add_argument("--json", help="write check stats JSON here")
    a = ap.parse_args()
    try:
        with open(a.proof) as f:
            proof = json.load(f)
        with open(a.expected) as f:
            expected = json.load(f)
    except Exception as e:
        print(f"NO: cannot load proof/expected: {e}")
        sys.exit(1)
    try:
        ok, reason, stats = verify(proof, expected)
    except Reject as e:
        print(f"NO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"NO: malformed proof: {e}")
        sys.exit(1)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(stats, f, indent=1)
    print("YES")
    sys.exit(0)


if __name__ == "__main__":
    main()