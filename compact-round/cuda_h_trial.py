#!/usr/bin/env python3
"""Compare the public Qwen Groth16 prover with CPU and CUDA H."""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import sys

from benchmark import ROOT, sha, timed
from key_format_trial import PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cuda_prover", "cpu_verifier", "pk", "output"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    os.environ["RAYON_NUM_THREADS"] = "2"
    statement_dir, proof_dir, pin, key_pin = PROFILES["rtx"]
    manifest = ROOT / "results" / statement_dir / "manifest.json"
    adapter = manifest.with_name("adapter.json")
    shape = ROOT / "results" / proof_dir / "shape.json"
    vk = shape.with_name("verifying.key")
    statement = [shape, manifest, pin, adapter, args.pk]
    verifier = [args.cpu_verifier, "prefix-verify", manifest, pin, adapter, vk, key_pin]
    saved = {}

    def review(label, proof):
        checked = timed(args.output, f"{label}-verify", [*verifier, proof])
        assert checked["reported"]["metrics"]["accepted"] and proof.stat().st_size == 128
        saved[str(proof.relative_to(args.output))] = sha(proof)

    resident = args.output / "resident"
    warm = timed(args.output, "paired-resident",
                 [args.cuda_prover, "prefix-bench-cuda-h", *statement, key_pin, resident])
    trials = warm["reported"]["metrics"]["trials"]
    assert len(trials) == 7 and [row["round"] for row in trials] == list(range(7))
    for row in trials:
        review(f"resident-{row['round']}", resident / f"{row['round']}.proof")
    warm["median_prove_seconds"] = {
        "cpu": median(row["cpu_prove_seconds"] for row in trials[1:]),
        "cuda": median(row["cuda"]["prove_seconds"] for row in trials[1:]),
    }
    print("Resident medians", warm["median_prove_seconds"], flush=True)

    fresh = {"cpu": [], "cuda": []}
    for pair in range(3):
        order = ("cpu", "cuda") if pair % 2 == 0 else ("cuda", "cpu")
        for backend in order:
            label = f"fresh-{pair + 1}-{backend}"
            proof = args.output / f"{label}.proof"
            command = ([args.cuda_prover, "prefix-prove-cuda-h", *statement, key_pin, proof]
                       if backend == "cuda"
                       else [args.cuda_prover, "prefix-prove-uncompressed", *statement, proof])
            fresh[backend].append(timed(args.output, label, command))
            review(label, proof)
            print(f"{label} verified", flush=True)
    bundle = manifest.with_name("bundle.json")
    transparent = []
    for i in range(3):
        recovered = args.output / f"transparent-{i}.json"
        transparent.append(timed(args.output, f"transparent-{i}", [
            sys.executable, "-B", ROOT / "round_verifier.py", manifest, bundle,
            "--manifest-sha256", pin, "--adapter", recovered,
        ]))
        assert json.loads(recovered.read_text()) == json.loads(adapter.read_text())
    report = {
        "manifest_sha256": pin, "verifying_key_sha256": key_pin, "proving_key_sha256": sha(args.pk),
        "cuda_prover_sha256": sha(args.cuda_prover), "cpu_verifier_sha256": sha(args.cpu_verifier),
        "rayon_threads": 2, "cpus": sorted(os.sched_getaffinity(0)), "proof_sha256": saved,
        "resident": warm, "fresh": fresh,
        "transparent": transparent,
        "sizes_bytes": {
            "compact_cached_key": manifest.stat().st_size + adapter.stat().st_size + 128,
            "compact_first_key": manifest.stat().st_size + adapter.stat().st_size + 128 + vk.stat().st_size,
            "transparent_package": manifest.stat().st_size + bundle.stat().st_size,
        },
        "median_fresh_process_seconds": {
            backend: median(row["process_seconds"] for row in rows) for backend, rows in fresh.items()
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
