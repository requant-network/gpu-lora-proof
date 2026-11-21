#!/usr/bin/env python3
"""Replay a bounded LoRA update to the final Qwen attention output projection."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from text_head_round import (
    LABELS, TARGET, V, make_manifest, quantize_features, read_inputs,
    read_raw_features, score, source_normalization,
)
from verify_text_head_origin import CPU_FILES, require_digest, sha256


STEPS = (0, -1, 1, -2, 2, -4, 4, -8, 8)
TENSOR = "blk.23.attn_output.weight"


def evaluate(head, rows):
    predictions = score(head, head["initial"], rows)
    return {
        "correct": sum(row["label"] == row["prediction"] for row in predictions),
        "squared_error": sum(
            (row["scores"][0] - value) ** 2 + (row["scores"][1] + value) ** 2
            for row in predictions
            for value in (TARGET if row["label"] == LABELS[0] else -TARGET,)
        ),
    }


def checked_run(command, env, output=subprocess.DEVNULL):
    result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        log = result.stderr.decode(errors="replace").splitlines()
        raise V.Reject(f"{Path(command[0]).name} failed: {log[-1] if log else result.returncode}")


def run(args):
    began = time.perf_counter()
    recipe, recipe_pin = V.read_json(args.recipe, V.MANIFEST_BYTES)
    require_digest(recipe_pin, args.recipe_sha256, "CPU recipe")
    if (recipe["schema"] != "qwen-head-feature-replay/v1" or recipe.get("profile") != "cpu"
            or set(recipe["runtime"]) != CPU_FILES):
        raise V.Reject("expected pinned CPU feature recipe")
    corpus_pin = sha256(args.corpus)
    require_digest(corpus_pin, recipe["corpus_sha256"], "corpus")
    require_digest(sha256(args.model), recipe["model_sha256"], "GGUF")
    library = args.library.resolve()
    for name, digest in recipe["runtime"].items():
        require_digest(sha256(library / name), digest, name)

    expected = V.read_json(args.certificate, V.MANIFEST_BYTES)[0] if args.certificate else None
    env = {"PATH": os.defpath, "LC_ALL": "C", "LD_LIBRARY_PATH": str(library),
           "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "3"}
    probe = [str(args.probe.resolve()), str(args.model.resolve()), str(library)]
    writer = str(args.writer.resolve())

    with tempfile.TemporaryDirectory(prefix="requant-qwen-projection-") as directory:
        root = Path(directory)

        def capture(path, output, adapter=None):
            command = [*probe, str(path), "--cpu"]
            if adapter is not None:
                command.append(str(adapter))
            with output.open("wb") as file:
                checked_run(command, env, output=file)

        baseline = root / "baseline.tsv"
        capture(args.corpus.resolve(), baseline)
        baseline_pin = sha256(baseline)
        require_digest(baseline_pin, recipe["features_sha256"], "baseline features")
        baseline_rows = read_inputs(args.corpus, baseline)
        head = make_manifest(baseline_rows, corpus_pin, baseline_pin)
        texts = [line.split("\t", 3) for line in args.corpus.read_text(encoding="ascii").splitlines()]
        raw = read_raw_features(texts, baseline)
        centers, scales = source_normalization(raw["source"])
        parts = {split: [row for row in texts if row[0] == split] for split in ("adapt", "heldout")}
        coefficients = [str(head["W"][i * V.D]) for i in range(V.D)]
        if not any(int(coefficient) for coefficient in coefficients):
            raise V.Reject("empty output direction")

        def evaluate_adapter(part, adapter, tag):
            corpus = root / f"{part}.tsv"
            corpus.write_bytes("".join("\t".join(row) + "\n" for row in parts[part]).encode("ascii"))
            features = root / f"features-{tag}.tsv"
            capture(corpus, features, adapter)
            captured = read_raw_features(parts[part], features)
            integer = quantize_features(captured, centers, scales)
            return {**evaluate(head, integer[part]), "features_sha256": sha256(features)}

        candidates = [{"step": 0, **evaluate(head, baseline_rows["adapt"])}]
        adapters = {}
        for step in STEPS[1:]:
            adapter = root / f"step-{step}.gguf"
            checked_run([writer, str(adapter), str(step), *coefficients], env)
            adapters[step] = adapter
            candidates.append({"step": step, "adapter_sha256": sha256(adapter),
                               **evaluate_adapter("adapt", adapter, str(step))})

        chosen = min(candidates, key=lambda row: (row["squared_error"], abs(row["step"]), row["step"]))
        before = evaluate(head, baseline_rows["heldout"])
        step = chosen["step"]
        after = evaluate_adapter("heldout", adapters[step], "heldout") if step else before
        released = bool(step and after["correct"] >= before["correct"]
                        and after["squared_error"] < before["squared_error"])
        certificate = {
            "schema": "qwen-projection-replay/v1",
            "cpu_recipe_sha256": recipe_pin,
            "target_tensor": TENSOR,
            "baseline_features_sha256": baseline_pin,
            "candidates": candidates,
            "selected_step": step,
            "heldout_before": before,
            "heldout_after": after,
            "release": released,
        }
        if expected is not None and V.canonical(expected) != V.canonical(certificate):
            raise V.Reject("projection certificate differs from replay")

        args.output.mkdir()
        V.save_json(args.output / "certificate.json", certificate)
        if released:
            with (args.output / "adapter.gguf").open("xb") as out:
                out.write(adapters[step].read_bytes())

    return {"certificate_sha256": sha256(args.output / "certificate.json"), "selected_step": step,
            "heldout_before": before, "heldout_after": after, "release": released,
            "replay_seconds": time.perf_counter() - began}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("writer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--certificate", type=Path)
    try:
        print(json.dumps(run(parser.parse_args())))
    except (V.Reject, KeyError, TypeError, ValueError, OSError, OverflowError,
            RecursionError, subprocess.TimeoutExpired) as error:
        print(f"NO: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
