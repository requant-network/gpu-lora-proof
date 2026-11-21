#!/usr/bin/env python3
"""Derive circuit wiring from the verified v2 contribution order."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract as C
import round_verifier as V


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("bundle")
    parser.add_argument("output")
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    manifest, pin = V.read_json(args.manifest, V.MANIFEST_BYTES)
    if pin != args.manifest_sha256 or manifest["schema"] != V.PREFIX_SCHEMA:
        raise V.Reject("expected pinned v2 manifest")
    bundle, _ = V.read_json(args.bundle, V.BUNDLE_BYTES)
    expected, _ = V.verify_round(manifest, pin, bundle)
    adapter = manifest["initial"]
    orders = []
    for index, phase in enumerate(bundle["phases"]):
        factor = manifest["phases"][index]["factor"]
        gradients = [C.b64_decode_i64(row["proof"]["leaves"]["d" + factor]["data_b64"])
                     for row in phase["contributions"]]
        total = [sum(values) for values in zip(*gradients, strict=True)]
        updated = V.sign_step(adapter[factor], factor, total)
        changed = [i for i, (a, b) in enumerate(zip(adapter[factor], updated, strict=True)) if a != b]
        orders.append(sorted(changed, key=lambda i: (-abs(total[i]), i)))
        adapter = V.choose_prefix(manifest["W"], manifest["phases"][index]["slots"],
                                  adapter, factor, updated, total)
    if adapter != expected:
        raise V.Reject("shape construction did not recover the certified adapter")
    V.save_json(args.output, {"orders": orders, "labels": V.top1(manifest["heldout"]["target"])})


if __name__ == "__main__":
    main()
