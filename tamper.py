#!/usr/bin/env python3
"""Negative-control (tamper) harness for gpu-lora-proof.

Each control returns a tampered proof that the independent verifier MUST
reject.  'Coherent' tampers recompute every SHA-256 leaf commitment, rederive
the Fiat-Shamir challenges, and recompute all Freivalds response vectors for
the tampered claims -- i.e. a maximally capable attacker that re-seals the
whole transcript.  Only the arithmetic truth (or the trusted input pins)
can then catch the tamper, which is exactly what these controls test.
"""
import array
import copy
import hashlib
from operator import mul

import contract as C


def _rehash_and_reseal(proof):
    """Recompute commitments, challenges and responses from current leaves."""
    par = proof["params"]
    d, r, b = par["d"], par["r"], par["b"]
    sh = C.shapes(d, r, b)
    leaves = {n: C.b64_decode_i64(proof["leaves"][n]["data_b64"])
              for n in C.LEAF_ORDER}
    hashes = {}
    for n in C.LEAF_ORDER:
        hashes[n] = C.leaf_hash(n, sh[n], array.array("q", leaves[n]).tobytes())
    proof["leaf_hashes"] = {n: hashes[n].hex() for n in C.LEAF_ORDER}
    trans = C.transcript(proof["step_id"], par, hashes)
    red = {n: [[v % C.P for v in row]
               for row in _rows(leaves[n], sh[n])] for n in C.LEAF_ORDER}
    tpose = {n: list(zip(*red[n])) for n in C.LEAF_ORDER}
    def mv(leaf, tr, vec):
        rws = tpose[leaf] if tr else red[leaf]
        return [sum(map(mul, row, vec)) % C.P for row in rws]
    resp = []
    for (i, rep, off, n) in C.challenge_layout(d, r, b)[0]:
        out, an, at, bn, bt, *_ = C.GEMM_SPEC[i]
        rv = C.challenge_vector(trans, i, rep, n)
        u = mv(bn, bt, rv)
        v = mv(an, at, u)
        w = mv(out, False, rv)
        resp.extend(u); resp.extend(v); resp.extend(w)
    proof["responses_b64"] = C.b64_encode_i64(resp)
    return proof


def _rows(flat, shape):
    rows, cols = shape
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]


def _set_leaf(proof, name, idx, delta):
    flat = C.b64_decode_i64(proof["leaves"][name]["data_b64"])
    flat[idx] += delta
    proof["leaves"][name]["data_b64"] = C.b64_encode_i64(flat)


def flip_dX_coherent(proof):
    p = copy.deepcopy(proof)
    _set_leaf(p, "dX", 0, 1)
    return _rehash_and_reseal(p), "exact combine dX = P + sR must fail"


def alter_W_coherent(proof):
    p = copy.deepcopy(proof)
    _set_leaf(p, "W", 0, 1)
    return _rehash_and_reseal(p), "trusted input pin for W must fail"


def wrong_y_coherent(proof):
    p = copy.deepcopy(proof)
    _set_leaf(p, "y", 0, 1)
    return _rehash_and_reseal(p), "exact combine y = H + sV must fail"


def wrong_dW_coherent(proof):
    p = copy.deepcopy(proof)
    _set_leaf(p, "dW", 0, 1)
    return _rehash_and_reseal(p), "Freivalds on dW = x^T g must fail"


def flip_U_coherent(proof):
    p = copy.deepcopy(proof)
    _set_leaf(p, "U", 0, 1)
    return _rehash_and_reseal(p), "Freivalds on U = x A^T must fail"


def truncated_responses(proof):
    p = copy.deepcopy(proof)
    del p["responses_b64"]
    return p, "missing response buffer must fail"


def truncated_leaf(proof):
    p = copy.deepcopy(proof)
    s = p["leaves"]["W"]["data_b64"]
    p["leaves"]["W"]["data_b64"] = s[:len(s) // 2]
    return p, "truncated leaf data must fail"


def wrong_step_id(proof):
    p = copy.deepcopy(proof)
    p["step_id"] = p["step_id"] + "-evil"
    return p, "step_id must match expected"


def wrong_scale(proof):
    p = copy.deepcopy(proof)
    p["params"]["s"] = p["params"]["s"] + 1
    return p, "params (scale s) must match expected"


def tamper_response_v(proof):
    p = copy.deepcopy(proof)
    resp = C.b64_decode_i64(p["responses_b64"])
    lay, _ = C.response_layout(p["params"]["d"], p["params"]["r"], p["params"]["b"])
    _, _, uo, vo, wo, k, m = lay[0]
    resp[vo] = (resp[vo] + 1) % C.P
    p["responses_b64"] = C.b64_encode_i64(resp)
    return p, "response binding (v vector) must fail"


def wrong_x_coherent(proof):
    p = copy.deepcopy(proof)
    flat = C.b64_decode_i64(p["leaves"]["x"]["data_b64"])
    flat[0] = flat[0] + 1 if flat[0] < C.X_MAX else flat[0] - 1
    p["leaves"]["x"]["data_b64"] = C.b64_encode_i64(flat)
    return _rehash_and_reseal(p), "trusted input pin for x must fail"


def stale_replay(proof_other_seed):
    """Present a proof from another seed against this seed's expected pins."""
    return copy.deepcopy(proof_other_seed), "foreign-seed proof must fail pins/params"


# Controls applicable to a single proof (stale_replay needs two proofs).
SINGLE_CONTROLS = [
    ("flip_dX_coherent", flip_dX_coherent),
    ("alter_W_coherent", alter_W_coherent),
    ("wrong_y_coherent", wrong_y_coherent),
    ("wrong_dW_coherent", wrong_dW_coherent),
    ("flip_U_coherent", flip_U_coherent),
    ("truncated_responses", truncated_responses),
    ("truncated_leaf", truncated_leaf),
    ("wrong_step_id", wrong_step_id),
    ("wrong_scale", wrong_scale),
    ("tamper_response_v", tamper_response_v),
    ("wrong_x_coherent", wrong_x_coherent),
]