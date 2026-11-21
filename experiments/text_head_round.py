#!/usr/bin/env python3
"""Verify a bounded LoRA round on features from a frozen Qwen model."""

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import round_experiment as experiment
import round_verifier as V


LABELS = ("sports", "weather")
SPLITS = ("source", "adapt", "heldout")
TARGET = 96


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(corpus, features):
    texts = [line.split("\t", 3) for line in corpus.read_text(encoding="ascii").splitlines()]
    counts = Counter((split, label) for split, _, label, _ in texts)
    if counts != Counter({(split, label): 8 for split in SPLITS for label in LABELS}):
        raise ValueError("expected balanced text splits")
    if len({identity for _, identity, _, _ in texts}) != len(texts) or len({text for _, _, _, text in texts}) != len(texts):
        raise ValueError("text or ID reused")

    header, *rows = features.read_text(encoding="ascii").splitlines()
    if header != "width\t896":
        raise ValueError("unexpected feature source width")
    records = {name: [] for name in SPLITS}
    for (split, identity, label, _), row in zip(texts, rows, strict=True):
        actual, tokens, *raw = row.split("\t")
        values = [float(v) for v in raw]
        if (identity != actual or len(values) != V.D or not 0 < int(tokens) <= 256
                or not all(math.isfinite(v) for v in values)):
            raise ValueError("invalid captured hidden row")
        records[split].append((identity, label, values))

    source = list(zip(*(row for _, _, row in records["source"]), strict=True))
    centers = [statistics.mean(column) for column in source]
    scales = [statistics.pstdev(column) or 1.0 for column in source]
    for split, items in records.items():
        records[split] = [
            (identity, label, [max(-16, min(16, round(4 * (v - center) / scale)))
                               for v, center, scale in zip(row, centers, scales, strict=True)])
            for identity, label, row in items
        ]
    return records


def make_manifest(records, corpus_pin, feature_pin):
    centroids = [[statistics.mean(col) for col in zip(*(r for _, label, r in records["source"] if label == key), strict=True)]
                 for key in LABELS]
    direction = [round((a - b) / 2) for a, b in zip(*centroids, strict=True)]
    weights = [v for direction_i in direction for v in (direction_i, -direction_i, *([0] * (V.D - 2)))]
    initial = {"A": [int(i == j * (V.D // V.R)) for j in range(V.R) for i in range(V.D)],
               "B": [0] * (V.D * V.R)}

    def batch(records):
        x, target = [], []
        for _, label, row in records:
            value = TARGET if label == LABELS[0] else -TARGET
            x.extend(row)
            target.extend((value, -value, *([0] * (V.D - 2))))
        return {"x": x, "target": target}

    phases = []
    for phase, factor in enumerate(V.FACTORS):
        slots = []
        for worker, name in enumerate(V.WORKERS):
            i = (phase * len(V.WORKERS) + worker) * V.B
            slots.append({"worker": name, **batch(records["adapt"][i:i + V.B])})
        phases.append({"factor": factor, "slots": slots})
    # The independent utility criterion uses the baseline's held-out error.
    max_prediction = V.D * 16 * 16 + V.SCALE * V.R * V.D * 16 * 2
    max_sse = V.EVAL_ROWS * V.D * (max_prediction + TARGET) ** 2
    return {"schema": V.SCHEMA, "id": f"qwen-head-{corpus_pin[:16]}-{feature_pin[:16]}", "W": weights,
            "initial": initial, "phases": phases, "heldout": batch(records["heldout"]), "max_heldout_sse": max_sse}


def score(manifest, adapter, records):
    predicted = V.predict(manifest["W"], adapter, [v for _, _, row in records for v in row])
    return [{"id": identity, "label": label, "scores": predicted[i * V.D:i * V.D + 2],
             "prediction": LABELS[predicted[i * V.D + 1] > predicted[i * V.D]]}
            for i, (identity, label, _) in enumerate(records)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lib")
    args = parser.parse_args()
    corpus_pin, feature_pin = sha(args.corpus), sha(args.features)
    records = read_inputs(args.corpus, args.features)
    manifest = make_manifest(records, corpus_pin, feature_pin)
    args.output.mkdir()
    manifest_path = args.output / "manifest.json"
    bundle_path = args.output / "bundle.json"
    adapter_path = args.output / "adapter.json"
    V.save_json(manifest_path, manifest)
    pin = sha(manifest_path)
    bundle, assembly = experiment.assemble(manifest_path, bundle_path, args.lib)
    verified = experiment.verify_cli(manifest_path, pin, bundle_path, adapter_path)
    adapter, _ = V.read_json(adapter_path, V.MANIFEST_BYTES)
    control = experiment.controls(manifest, pin, bundle)
    prior = score(manifest, manifest["initial"], records["heldout"])
    final = score(manifest, adapter, records["heldout"])
    before = sum(row["label"] == row["prediction"] for row in prior)
    after = sum(row["label"] == row["prediction"] for row in final)
    report = {"corpus_sha256": corpus_pin, "features_sha256": feature_pin, "manifest_sha256": pin,
              "adapter_sha256": sha(adapter_path), "bundle_sha256": sha(bundle_path), "assembly": assembly,
              "verification": verified, "rejected_controls": control, "initial_correct": before,
              "final_correct": after, "heldout_initial": prior, "heldout_final": final,
              "utility_passed": after > before and verified["final_sse"] <= verified["initial_sse"]}
    V.save_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
