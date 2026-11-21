#!/usr/bin/env python3
"""Verify a bounded LoRA round on features from a frozen Qwen model."""

import argparse
from collections import Counter
import copy
import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract as C
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

    records = read_raw_features(texts, features)
    centers, scales = source_normalization(records["source"])
    return quantize_features(records, centers, scales)


def read_raw_features(texts, features):
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

    return records


def source_normalization(source_rows):
    source = list(zip(*(row for _, _, row in source_rows), strict=True))
    return ([statistics.mean(column) for column in source],
            [statistics.pstdev(column) or 1.0 for column in source])


def quantize_features(records, centers, scales):
    return {
        split: [(identity, label, [max(-16, min(16, round(4 * (v - center) / scale)))
                                  for v, center, scale in zip(row, centers, scales, strict=True)])
                for identity, label, row in items]
        for split, items in records.items()
    }


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


def combined_limits(baseline_sse, baseline_correct):
    if baseline_sse <= 0:
        raise ValueError("baseline error leaves no room for improvement")
    return {"max_heldout_sse": baseline_sse - 1, "min_heldout_correct": baseline_correct}


def phase_diagnostics(manifest, pin, bundle, records, verified_adapter):
    adapter = manifest["initial"]
    snapshots = [("initial", adapter, 0)]
    for i, phase in enumerate(bundle["phases"]):
        factor = V.FACTORS[i]
        prior = adapter[factor]
        adapter = V.advance(manifest, pin, i, adapter, phase["contributions"])
        changed = sum(a != b for a, b in zip(prior, adapter[factor], strict=True))
        snapshots.append((f"after_{factor}", adapter, changed))
    if adapter != verified_adapter:
        raise RuntimeError("diagnostic replay disagrees with independent verifier")

    splits = {"source": records["source"], "adapt_B": records["adapt"][:8],
              "adapt_A": records["adapt"][8:], "heldout": records["heldout"]}
    result = {}
    for stage, snapshot, changed in snapshots:
        summary = {"changed_coordinates": changed}
        for split, examples in splits.items():
            rows = score(manifest, snapshot, examples)
            error = 0
            for row in rows:
                expected = TARGET if row["label"] == LABELS[0] else -TARGET
                error += (row["scores"][0] - expected)**2 + (row["scores"][1] + expected)**2
            summary[split] = {"correct": sum(r["label"] == r["prediction"] for r in rows),
                              "total": len(rows), "topic_scores_sse": error}
        result[stage] = summary
    return result


def check_quality_gate(manifest, output, limits, candidate_sha, refusal=None):
    output.mkdir()
    manifest_path, bundle_path, adapter_path = (output / name for name in ("manifest.json", "bundle.json", "adapter.json"))
    V.save_json(manifest_path, {**manifest, **limits})
    pin = sha(manifest_path)
    experiment.assemble(manifest_path, bundle_path, lib=None)
    report = {"manifest_sha256": pin, "bundle_sha256": sha(bundle_path), "sse_ceiling": limits["max_heldout_sse"]}
    if "min_heldout_correct" in limits:
        report["min_heldout_correct"] = limits["min_heldout_correct"]
    try:
        experiment.verify_cli(manifest_path, pin, bundle_path, adapter_path)
    except subprocess.CalledProcessError as error:
        reason = error.stderr.decode("ascii").strip()
        if refusal is None or error.returncode != 1 or reason != f"NO: {refusal}" or adapter_path.exists():
            raise
        return {**report, "exit_code": error.returncode, "reason": reason, "adapter_written": False}
    if refusal is not None or sha(adapter_path) != candidate_sha:
        raise RuntimeError("quality gate released an unexpected adapter")
    return {**report, "exit_code": 0, "adapter_written": True, "adapter_sha256": candidate_sha}


def check_full_step_rejection(manifest, pin, bundle):
    adapter = manifest["initial"]
    for i, (factor, phase) in enumerate(zip(V.FACTORS, bundle["phases"], strict=True)):
        records = phase["contributions"]
        expected = V.advance(manifest, pin, i, adapter, records)
        gradients = [C.b64_decode_i64(record["proof"]["leaves"]["d" + factor]["data_b64"]) for record in records]
        totals = [sum(terms) for terms in zip(*gradients, strict=True)]
        full = {**adapter, factor: V.sign_step(adapter[factor], factor, totals)}
        if full != expected:
            altered = copy.deepcopy(bundle)
            phases = altered["phases"]
            phases[i]["next"] = V.adapter_digest(pin, full, V.PREFIX_SCHEMA)
            altered["merkle_root"] = V.merkle_root(phases, V.PREFIX_SCHEMA)
            try:
                V.verify_round(manifest, pin, altered)
            except V.Reject as error:
                if str(error) != "incorrect optimizer transition":
                    raise
                return {"factor": factor, "reason": str(error)}
            raise RuntimeError("accepted a full step in place of the selected prefix")
        adapter = expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lib")
    parser.add_argument("--prefix", action="store_true", help="use the training-loss-selected round schema")
    args = parser.parse_args()
    corpus_pin, feature_pin = sha(args.corpus), sha(args.features)
    records = read_inputs(args.corpus, args.features)
    manifest = make_manifest(records, corpus_pin, feature_pin)
    if args.prefix:
        manifest["schema"] = V.PREFIX_SCHEMA
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
    initial_sse, final_sse = verified["initial_sse"], verified["final_sse"]
    if args.prefix:
        utility = after >= before and final_sse < initial_sse
    else:
        utility = after > before and final_sse <= initial_sse
    report = {"corpus_sha256": corpus_pin, "features_sha256": feature_pin, "manifest_sha256": pin,
              "adapter_sha256": sha(adapter_path), "bundle_sha256": sha(bundle_path), "assembly": assembly,
              "verification": verified, "rejected_controls": control, "initial_correct": before,
              "final_correct": after, "heldout_initial": prior, "heldout_final": final,
              "phase_diagnostics": phase_diagnostics(manifest, pin, bundle, records, adapter),
              "utility_passed": utility}
    if report["phase_diagnostics"]["after_A"]["heldout"]["topic_scores_sse"] != verified["final_sse"]:
        raise RuntimeError("unexpected nonzero scores outside the two topic columns")
    if args.prefix or final_sse > initial_sse:
        refusal = "held-out quality gate failed" if final_sse > initial_sse else None
        report["quality_gate_control"] = check_quality_gate(manifest, args.output / "quality-gate",
                                                           {"max_heldout_sse": initial_sse},
                                                           report["adapter_sha256"], refusal)
    if args.prefix:
        report["full_step_control"] = check_full_step_rejection(manifest, pin, bundle)
        if initial_sse > 0:
            refusal = None
            if final_sse >= initial_sse:
                refusal = "held-out quality gate failed"
            elif after < before:
                refusal = "held-out correctness gate failed"
            limits = combined_limits(initial_sse, before)
            report["combined_quality_gate_control"] = check_quality_gate(
                manifest, args.output / "combined-gate", limits, report["adapter_sha256"], refusal)
    V.save_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
