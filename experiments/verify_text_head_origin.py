#!/usr/bin/env python3
"""Recreate the Qwen features, bind the manifest, then verify an adapter release."""

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from text_head_round import V, combined_limits, make_manifest, read_inputs


CPU_FILES = {"libllama.so.0", "libggml.so.0", "libggml-base.so.0", "libggml-cpu-haswell.so"}
RTX_FILES = CPU_FILES | {
    "cuda_v13/libggml-cuda.so", "cuda_v13/libcudart.so.13",
    "cuda_v13/libcublas.so.13", "cuda_v13/libcublasLt.so.13",
}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require_digest(actual, expected, label):
    if not hmac.compare_digest(actual, expected):
        raise V.Reject(f"{label} SHA-256 mismatch")


def reconstruct(corpus, features, corpus_pin, feature_pin):
    manifest = make_manifest(read_inputs(corpus, features), corpus_pin, feature_pin)
    manifest["schema"] = V.PREFIX_SCHEMA
    heldout = manifest["heldout"]
    initial = V.predict(manifest["W"], manifest["initial"], heldout["x"])
    sse = V.squared_error(initial, heldout["target"])
    correct = sum(a == b for a, b in zip(V.top1(initial), V.top1(heldout["target"]), strict=True))
    return {**manifest, **combined_limits(sse, correct)}


def verify(args):
    started = time.perf_counter()
    recipe, pin = V.read_json(args.recipe, V.MANIFEST_BYTES)
    require_digest(pin, args.recipe_sha256, "replay recipe")
    if recipe["schema"] != "qwen-head-feature-replay/v1":
        raise V.Reject("unexpected recipe schema")
    profile = recipe.get("profile", "rtx")
    runtime = {"rtx": RTX_FILES, "cpu": CPU_FILES}
    if profile not in runtime or set(recipe["runtime"]) != runtime[profile]:
        raise V.Reject("unexpected replay profile")
    if profile == "rtx" and not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu or ""):
        raise V.Reject("select the RTX by its GPU UUID")
    if profile == "cpu" and args.gpu is not None:
        raise V.Reject("CPU replay does not take a GPU UUID")

    corpus_pin = sha256(args.corpus)
    require_digest(corpus_pin, recipe["corpus_sha256"], "corpus")
    require_digest(sha256(args.model), recipe["model_sha256"], "GGUF")
    library = args.library.resolve()
    for name, expected in recipe["runtime"].items():
        require_digest(sha256(library / name), expected, name)

    bundle, _ = V.read_json(args.bundle, V.BUNDLE_BYTES)
    require_digest(bundle["manifest_sha256"], recipe["manifest_sha256"], "bundle manifest")
    env = {"PATH": os.defpath, "LC_ALL": "C", "LD_LIBRARY_PATH": str(library)}
    if profile == "rtx":
        capture_flag = "--pinned-rtx"
        env["LD_LIBRARY_PATH"] += f":{library / 'cuda_v13'}"
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        capture_flag = "--cpu"
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["OMP_NUM_THREADS"] = "3"
    pinned = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="requant-feature-replay-") as path:
        features = Path(path) / "features.tsv"
        with features.open("wb") as output:
            captured = subprocess.run(
                [str(args.probe.resolve()), str(args.model.resolve()), str(library),
                 str(args.corpus.resolve()), capture_flag],
                env=env,
                stdout=output, stderr=subprocess.PIPE, timeout=120, check=False,
            )
        if captured.returncode:
            error = captured.stderr.decode(errors="replace").splitlines()
            raise V.Reject(f"capture failed ({captured.returncode}): {error[-1] if error else 'no diagnostics'}")
        captured_at = time.perf_counter()
        feature_pin = sha256(features)
        require_digest(feature_pin, recipe["features_sha256"], "captured features")
        manifest = reconstruct(args.corpus, features, corpus_pin, feature_pin)
        manifest_pin = hashlib.sha256(V.canonical(manifest) + b"\n").hexdigest()
        require_digest(manifest_pin, recipe["manifest_sha256"], "reconstructed manifest")
        adapter, report = V.verify_round(manifest, manifest_pin, bundle)
    V.save_json(args.adapter, adapter)
    return {
        **report,
        "recipe_sha256": pin,
        "corpus_sha256": corpus_pin,
        "features_sha256": feature_pin,
        "manifest_sha256": manifest_pin,
        "adapter_sha256": sha256(args.adapter),
        "pin_check_seconds": pinned - started,
        "capture_seconds": captured_at - pinned,
        "verify_wall_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("probe", type=Path, help="locally built, trusted capture executable")
    parser.add_argument("adapter", type=Path, help="fresh output path")
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--gpu", help="GPU UUID required by an RTX recipe")
    try:
        report = verify(parser.parse_args())
    except (V.Reject, KeyError, TypeError, ValueError, OSError, OverflowError,
            RecursionError, subprocess.TimeoutExpired) as exc:
        print(f"NO: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
