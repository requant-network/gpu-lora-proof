#!/usr/bin/env python3
"""Compare checked proving-key encodings on the same public Qwen circuits."""

import argparse
import json
import os
from pathlib import Path
from statistics import median

from benchmark import ROOT, sha, timed


PROFILES = {
    "rtx": (
        "text-head-v2/mixed-vocabulary", "compact-prefix",
        "d200e82458d79b44fc577480ba33b1e7fe79b6e935328bd7812cfdd05763c34a",
        "ef6ff88719ce916f2fc86a70e79de8e90470d4b0e534c5782ae9dbde8d8dd38d",
    ),
    "cpu": (
        "text-head-origin/cpu", "compact-prefix/cpu",
        "ccb4d88429a4e2fce34b1f1a3d7cd88f1755f3ab6aaf9d91262ad0a52eb54263",
        "082d83afa36e7d5867f93bc6e5d98a86bad2ac5c742cf0b84b584245b84970cd",
    ),
}
MSM_PHASES = ("Compute A", "Compute B in G1", "Compute B in G2", "Compute C")
FORMATS = ("compressed", "uncompressed")


def trial(out, executable, name, original):
    statement_dir, proof_dir, pin, key_pin = PROFILES[name]
    result_dir = ROOT / "results"
    manifest = result_dir / statement_dir / "manifest.json"
    adapter = manifest.with_name("adapter.json")
    shape = result_dir / proof_dir / "shape.json"
    vk = shape.with_name("verifying.key")
    original_proof = shape.with_name("round.proof")
    statement = (manifest, pin, adapter)
    verify = (executable, "prefix-verify", *statement, vk, key_pin)
    uncompressed = out / (name + "-uncompressed.key")
    baseline = timed(out, name + "-prior-verify", [*verify, original_proof])
    assert baseline["reported"]["metrics"]["accepted"]
    conversion = timed(out, name + "-convert", [executable, "proving-key-uncompress", original, uncompressed])
    assert original.stat().st_size < uncompressed.stat().st_size
    assert conversion["reported"]["metrics"]["proving_key_bytes"] == uncompressed.stat().st_size
    commands = {
        "compressed": (executable, "prefix-prove", shape, *statement, original),
        "uncompressed": (executable, "prefix-prove-uncompressed", shape, *statement, uncompressed),
    }
    runs = {encoding: [] for encoding in FORMATS}
    for index in range(4):
        order = FORMATS if index % 2 == 0 else FORMATS[::-1]
        for encoding in order:
            label = f"{name}-{index + 1}-{encoding}"
            proof = out / (label + ".proof")
            timing = timed(out, label + "-prove", [*commands[encoding], proof])
            checked = timed(out, label + "-verify", [*verify, proof])
            assert checked["reported"]["metrics"]["accepted"] and proof.stat().st_size == 128
            timing["proof_sha256"] = sha(proof)
            timing["msm_phase_seconds"] = sum(timing["trace_seconds"][phase] for phase in MSM_PHASES)
            assert timing["msm_phase_seconds"] <= timing["reported"]["metrics"]["prove_seconds"]
            runs[encoding].append(timing)
            print(label, "accepted", flush=True)
    controls = {}
    if name == "rtx":
        # The first uncompressed BN254 G1 point is the verification key's nonzero alpha.
        # A zero Y coordinate with no infinity flag cannot be a valid G1 point.
        bad = out / "off-curve.key"
        raw = bytearray(uncompressed.read_bytes())
        raw[32:64] = bytes(32)
        bad.write_bytes(raw)
        output = out / "rejected.proof"
        controls["off_curve_key"] = timed(out, "off-curve", [*commands["uncompressed"][:-1], bad, output],
                                           reject="invalid")
        assert not output.exists()
        raw = bytearray(uncompressed.read_bytes())
        bad = out / "appended.key"
        bad.write_bytes(raw + b"\0")
        controls["appended_key_data"] = timed(out, "appended-key", [*commands["uncompressed"][:-1], bad, output],
                                               reject="extra bytes after canonical encoding")
        assert not output.exists()
    return {
        "manifest_sha256": pin, "key_sha256": key_pin,
        "formats": {
            encoding: {"bytes": path.stat().st_size, "sha256": sha(path), "runs": runs[encoding],
                       "median_process_seconds": median(run["process_seconds"] for run in runs[encoding]),
                       "median_key_load_seconds": median(run["reported"]["metrics"]["key_load_seconds"] for run in runs[encoding]),
                       "median_msm_phase_seconds": median(run["msm_phase_seconds"] for run in runs[encoding])}
            for encoding, path in zip(FORMATS, (original, uncompressed))
        },
        "conversion": conversion, "controls": controls,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "rtx_key", "cpu_key", "output"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    os.environ["RAYON_NUM_THREADS"] = "2"
    report = {"executable_sha256": sha(args.executable), "cpus": sorted(os.sched_getaffinity(0)), "rayon_threads": 2,
              "profiles": {name: trial(args.output, args.executable, name, key)
                           for name, key in (("rtx", args.rtx_key), ("cpu", args.cpu_key))}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
