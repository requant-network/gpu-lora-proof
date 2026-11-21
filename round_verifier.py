#!/usr/bin/env python3
"""Verify a bounded two-phase integer LoRA round and recover its adapter."""

import argparse
import array
import hashlib
import hmac
import json
from pathlib import Path
import sys
import time

import contract as C
from verifier import Reject, verify as verify_step


SCHEMA = "gpu-lora-round/v1"
PREFIX_SCHEMA = "gpu-lora-round/v2"
D, R, B, SCALE = 32, 4, 4, C.S_DEFAULT
FACTORS = ("B", "A")
WORKERS = ("worker-a", "worker-b")
EVAL_ROWS = 16
SHAPES = C.shapes(D, R, B)
MANIFEST_BYTES = 64 << 10
BUNDLE_BYTES = 1 << 20


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path, limit):
    def distinct(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise Reject("duplicate JSON key")
        return value

    with Path(path).open("rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise Reject("JSON exceeds round size limit")
    value = json.loads(raw, object_pairs_hook=distinct)
    if not isinstance(value, dict):
        raise Reject("round JSON must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    with Path(path).open("xb") as f:
        f.write(canonical(value) + b"\n")


def bounded_ints(values, n, bound):
    return (isinstance(values, list) and len(values) == n
            and all(type(v) is int and abs(v) <= bound for v in values))


def validate_manifest(m):
    if (m["schema"] not in (SCHEMA, PREFIX_SCHEMA) or not isinstance(m["id"], str)
            or not 1 <= len(m["id"]) <= 128 or type(m["max_heldout_sse"]) is not int
            or m["max_heldout_sse"] < 0):
        raise Reject("invalid round manifest")
    if (not bounded_ints(m["W"], D * D, C.W_MAX)
            or not bounded_ints(m["initial"]["A"], R * D, C.A_MAX)
            or not bounded_ints(m["initial"]["B"], D * R, C.B_MAX)
            or set(m["initial"]) != {"A", "B"}):
        raise Reject("invalid base weights or initial adapter")
    phases = m["phases"]
    if [p["factor"] for p in phases] != list(FACTORS):
        raise Reject("incorrect optimizer schedule")
    trained = set()
    for phase in phases:
        if [slot["worker"] for slot in phase["slots"]] != list(WORKERS):
            raise Reject("incorrect contribution slots")
        for slot in phase["slots"]:
            if not (bounded_ints(slot["x"], B * D, C.X_MAX)
                    and bounded_ints(slot["target"], B * D, C.G_MAX)):
                raise Reject("invalid training batch")
            trained.update(tuple(slot["x"][i:i+D]) for i in range(0, B * D, D))
    x, y = m["heldout"]["x"], m["heldout"]["target"]
    if not bounded_ints(x, EVAL_ROWS * D, C.X_MAX) or not bounded_ints(y, EVAL_ROWS * D, C.G_MAX):
        raise Reject("invalid held-out data")
    heldout = {tuple(x[i:i+D]) for i in range(0, len(x), D)}
    if len(heldout) != EVAL_ROWS or heldout & trained:
        raise Reject("evaluation inputs repeat or overlap training")
    if "min_heldout_correct" in m:
        floor = m["min_heldout_correct"]
        if m["schema"] != PREFIX_SCHEMA or type(floor) is not int or not 0 <= floor <= EVAL_ROWS:
            raise Reject("invalid held-out correctness floor")
        for i in range(0, len(y), D):
            row = y[i:i+D]
            if row.count(max(row)) != 1:
                raise Reject("held-out classification target needs a unique maximum")


def predict(weights, adapter, features):
    """Exact row-major Y = X W + scale (X A^T) B^T."""
    result = []
    a, b = adapter["A"], adapter["B"]
    for start in range(0, len(features), D):
        x = features[start:start+D]
        projected = [sum(x[i] * a[j*D+i] for i in range(D)) for j in range(R)]
        for out in range(D):
            base = sum(x[i] * weights[i*D+out] for i in range(D))
            delta = sum(projected[j] * b[out*R+j] for j in range(R))
            result.append(base + SCALE * delta)
    return result


def residual(predicted, target):
    return [max(-C.G_MAX, min(C.G_MAX, y - t)) for y, t in zip(predicted, target, strict=True)]


def squared_error(predicted, target):
    return sum((y - t)**2 for y, t in zip(predicted, target, strict=True))


def top1(values):
    winners = []
    for start in range(0, len(values), D):
        row = values[start:start + D]
        winners.append(max(range(D), key=row.__getitem__))
    return winners


def slot_seed(manifest_pin, phase, worker, schema=SCHEMA):
    label = canonical([schema, manifest_pin, phase, worker])
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big")


def adapter_digest(manifest_pin, adapter, schema=SCHEMA):
    return digest([schema, manifest_pin, "adapter", adapter])


def expected_step(manifest, phase, slot, manifest_pin, adapter, proof):
    seed = slot_seed(manifest_pin, phase, slot["worker"], manifest["schema"])
    predicted = C.b64_decode_i64(proof["leaves"]["y"]["data_b64"])
    if len(predicted) != B * D:
        raise Reject("incorrect prediction length")
    trusted = {"W": manifest["W"], **adapter, "x": slot["x"],
               "g": residual(predicted, slot["target"])}
    return {
        "step_id": C.step_id(D, R, B, SCALE, seed),
        "params": C.params_dict(D, R, B, SCALE, seed),
        "input_hashes": {
            k: C.leaf_hash(k, SHAPES[k], array.array("q", v).tobytes()).hex()
            for k, v in trusted.items()
        },
    }


def advance(manifest, manifest_pin, phase, adapter, contributions):
    plan = manifest["phases"][phase]
    slots = plan["slots"]
    if len(contributions) != len(slots):
        raise Reject("missing or extra contribution")
    if [c["worker"] for c in contributions] != [s["worker"] for s in slots]:
        raise Reject("duplicate or out-of-order contribution")
    factor = plan["factor"]
    gradients = []
    for slot, contribution in zip(slots, contributions, strict=True):
        proof = contribution["proof"]
        expected = expected_step(manifest, phase, slot, manifest_pin, adapter, proof)
        verify_step(proof, expected)
        gradients.append(C.b64_decode_i64(proof["leaves"]["d" + factor]["data_b64"]))
    totals = [sum(terms) for terms in zip(*gradients, strict=True)]
    updated = sign_step(adapter[factor], factor, totals)
    if manifest["schema"] == PREFIX_SCHEMA:
        return choose_prefix(manifest["W"], slots, adapter, factor, updated, totals)
    return {**adapter, factor: updated}


def sign_step(values, factor, totals):
    limit = C.A_MAX if factor == "A" else C.B_MAX
    updated = []
    for old, gradient in zip(values, totals, strict=True):
        direction = (gradient > 0) - (gradient < 0)
        updated.append(max(-limit, min(limit, old - direction)))
    return updated


def choose_prefix(weights, slots, adapter, factor, updated, totals):
    features = [value for slot in slots for value in slot["x"]]
    targets = [value for slot in slots for value in slot["target"]]
    old = adapter[factor]
    changed = [i for i, (previous, new) in enumerate(zip(old, updated, strict=True)) if previous != new]
    ordered = sorted(changed, key=lambda i: (-abs(totals[i]), i))
    best = old
    best_error = squared_error(predict(weights, adapter, features), targets)
    working = old[:]
    candidate = {**adapter, factor: working}
    for i in ordered:
        working[i] = updated[i]
        error = squared_error(predict(weights, candidate, features), targets)
        if error < best_error:
            best_error = error
            best = working[:]
    return {**adapter, factor: best}


def merkle_root(phases, schema=SCHEMA):
    leaves = [
        hashlib.sha256(b"\x00" + canonical(
            [schema, i, phase["previous"], record, phase["next"]]
        )).digest()
        for i, phase in enumerate(phases)
        for record in phase["contributions"]
    ]
    if len(leaves) != len(FACTORS) * len(WORKERS):
        raise Reject("invalid Merkle leaf count")
    while len(leaves) > 1:
        leaves = [hashlib.sha256(b"\x01" + left + right).digest()
                  for left, right in zip(leaves[::2], leaves[1::2], strict=True)]
    return leaves[0].hex()


def verify_round(manifest, manifest_pin, bundle):
    validate_manifest(manifest)
    schema = manifest["schema"]
    if bundle["schema"] != schema or bundle["manifest_sha256"] != manifest_pin:
        raise Reject("round manifest pin mismatch")
    phases = bundle["phases"]
    if len(phases) != len(manifest["phases"]):
        raise Reject("missing or extra phase")
    adapter = manifest["initial"]
    for i, phase in enumerate(phases):
        if phase["previous"] != adapter_digest(manifest_pin, adapter, schema):
            raise Reject("incorrect phase input adapter")
        adapter = advance(manifest, manifest_pin, i, adapter, phase["contributions"])
        if phase["next"] != adapter_digest(manifest_pin, adapter, schema):
            raise Reject("incorrect optimizer transition")
    if bundle["merkle_root"] != merkle_root(phases, schema):
        raise Reject("incorrect Merkle root")
    if bundle["final_adapter"] != adapter_digest(manifest_pin, adapter, schema):
        raise Reject("incorrect final adapter")
    heldout = manifest["heldout"]
    initial = predict(manifest["W"], manifest["initial"], heldout["x"])
    final = predict(manifest["W"], adapter, heldout["x"])
    targets = heldout["target"]
    initial_sse = squared_error(initial, targets)
    final_sse = squared_error(final, targets)
    if final_sse > manifest["max_heldout_sse"]:
        raise Reject("held-out quality gate failed")
    if "min_heldout_correct" in manifest:
        correct = sum(actual == wanted for actual, wanted in zip(top1(final), top1(targets), strict=True))
        if correct < manifest["min_heldout_correct"]:
            raise Reject("held-out correctness gate failed")
    return adapter, {"merkle_root": bundle["merkle_root"], "final_adapter": bundle["final_adapter"],
                     "initial_sse": initial_sse, "final_sse": final_sse, "heldout_rows": EVAL_ROWS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("bundle")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        manifest, pin = read_json(args.manifest, MANIFEST_BYTES)
        if not hmac.compare_digest(pin, args.manifest_sha256):
            raise Reject("manifest SHA-256 mismatch")
        bundle, _ = read_json(args.bundle, BUNDLE_BYTES)
        adapter, report = verify_round(manifest, pin, bundle)
        if args.adapter:
            save_json(args.adapter, adapter)
    except (Reject, KeyError, TypeError, ValueError, OSError, OverflowError, RecursionError) as exc:
        print(f"NO: {exc}", file=sys.stderr)
        return 1
    report["verify_wall_seconds"] = time.perf_counter() - start
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
