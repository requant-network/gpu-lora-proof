#!/usr/bin/env python3
"""Measure the same fixed LoRA round with replay, Freivalds and Groth16."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "results" / "round-v1"
PINS = {
    20261121: "c645e19ef587a2b582ceba48719de32032e5584d33eacdbcf33b0db39421fe65",
    90817: "95755b07cac5a522df5f94cbc721870ebf957c7a42944292fae197a6b0ade829",
}
TRACE = re.compile(r"End:\s+(.+?)\s+\.+\s*(\d+(?:\.\d+)?)(ns|µs|ms|s)$")
SCALE = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(out, label, args, *, reject=None, timeout=90):
    rss = out / (label + ".rss")
    start = time.perf_counter()
    proc = subprocess.run(["/usr/bin/time", "-f", "%M", "-o", str(rss), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - start
    (out / (label + ".log")).write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if reject is not None:
        if proc.returncode != 1 or reject not in proc.stderr:
            raise RuntimeError(f"wrong control result for {label}: {proc.returncode} {proc.stderr}")
        return proc.stderr.strip()
    if proc.returncode:
        raise RuntimeError(f"{label} failed: {proc.stderr}")
    lines = proc.stdout.splitlines()
    report = next((json.loads(line.removeprefix("RESULT ")) for line in lines
                   if line.startswith("RESULT ")), None)
    if report is None:
        report = json.loads(lines[-1])
    timer = {}
    for line in lines:
        match = TRACE.search(line)
        if match:
            name, value, unit = match.groups()
            timer[name] = float(value) * SCALE[unit]
    if label.endswith("-prove") and not all(name in timer for name in (
        "Constraint synthesis", "R1CS to QAP witness map", "Compute C", "Compute A", "Compute B in G1", "Compute B in G2",
    )):
        raise RuntimeError(f"{label}: incomplete prover profile")
    return {"process_seconds": wall, "max_rss_kib": int(rss.read_text()), "reported": report, "trace_seconds": timer}


def controls(out, bin_path, manifest, pin, adapter, vk, key_pin, proof):
    def verify(label, path=manifest, expected=pin, proposed=adapter, key=key_pin, p=proof, *, refusal):
        return timed(out, label, [bin_path, "verify", path, expected, proposed, vk, key, p], reject=refusal)

    base = json.loads(manifest.read_text())
    changed = out / f"{manifest.parent.name}-changed-input.json"
    base["W"][0] += 1
    changed.write_text(json.dumps(base))
    binding = out / f"{manifest.parent.name}-changed-id.json"
    base = json.loads(manifest.read_text())
    base["id"] += "-changed"
    binding.write_text(json.dumps(base))
    alternate = out / f"{manifest.parent.name}-changed-adapter.json"
    a = json.loads(adapter.read_text())
    a["A"][0] += 1
    alternate.write_text(json.dumps(a))
    appended = out / f"{manifest.parent.name}-extra-proof.bin"
    appended.write_bytes(proof.read_bytes() + b"\x00")
    name = manifest.parent.name
    groth = "Groth16 rejected"
    return {
        "wrong_manifest_pin": verify(name+"-wrong-pin", expected="0"*64, refusal="manifest SHA-256 mismatch"),
        "changed_weight": verify(name+"-changed-weight", changed, sha(changed), refusal=groth),
        "changed_manifest_identity": verify(name+"-changed-id", binding, sha(binding), refusal=groth),
        "changed_adapter": verify(name+"-changed-output", proposed=alternate, refusal=groth),
        "wrong_key_pin": verify(name+"-wrong-key", key="0"*64, refusal="verifying key SHA-256 mismatch"),
        "appended_proof_data": verify(name+"-extra-proof", p=appended, refusal="extra bytes after canonical encoding"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=lambda p: Path(p).resolve())
    parser.add_argument("output", type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    out, bin_path = args.output, args.executable
    out.mkdir(mode=0o700)
    os.environ["RAYON_NUM_THREADS"] = "2"
    shape = timed(out, "shape", [bin_path, "shape"])
    setup = timed(out, "setup", [bin_path, "setup", out / "keys"])
    pk, vk = (out / "keys" / f"{key}.key" for key in ("proving", "verifying"))
    key_pin = sha(vk)
    build = ("import json,sys;from pathlib import Path;from round_experiment import assemble;"
             "_,r=assemble(Path(sys.argv[1]),Path(sys.argv[2]),None);print(json.dumps(r))")
    rows = []
    for seed, pin in PINS.items():
        fixture = FIXTURES / str(seed)
        manifest, adapter, original = (fixture / name for name in ("manifest.json", "adapter.json", "mixed.json"))
        if sha(manifest) != pin:
            raise RuntimeError("trusted manifest has changed")
        trials = []
        for attempt in range(3):
            tag = f"{seed}-{attempt}"
            proof = out / f"{tag}.proof"
            certificate = out / f"{tag}.json"
            recovered = out / f"{tag}.adapter"
            trial = {"replay": timed(out, tag+"-replay", [bin_path, "replay", manifest, pin, adapter]),
                     "transparent_build": timed(out, tag+"-certificate-assemble",
                         [sys.executable, "-B", "-c", build, manifest, certificate])}
            if sha(certificate) != sha(original):
                raise RuntimeError("transparent CPU rebuild changed the retained round")
            trial["transparent_verify"] = timed(out, tag+"-certificate-verify",
                [sys.executable, "-B", ROOT / "round_verifier.py", manifest, certificate,
                 "--manifest-sha256", pin, "--adapter", recovered])
            if recovered.read_bytes() != adapter.read_bytes():
                raise RuntimeError("transparent verification did not recover the known adapter")
            trial["prove"] = timed(out, tag+"-prove", [bin_path, "prove", manifest, pin, adapter, pk, proof])
            trial["verify"] = timed(out, tag+"-verify", [bin_path, "verify", manifest, pin, adapter, vk, key_pin, proof])
            trials.append(trial)
            print(f"{tag}: transparent and compact proof verified", flush=True)
        retained = out / f"{seed}.proof"
        retained.write_bytes((out / f"{seed}-0.proof").read_bytes())
        rows.append({
            "seed": seed, "manifest_sha256": pin, "adapter_sha256": sha(adapter), "proof_sha256": sha(retained),
            "manifest_bytes": manifest.stat().st_size, "adapter_bytes": adapter.stat().st_size,
            "transparent_bundle_bytes": original.stat().st_size, "compact_proof_bytes": retained.stat().st_size,
            "trials": trials, "controls": controls(out, bin_path, manifest, pin, adapter, vk, key_pin, retained),
        })
    sources = ("Cargo.toml", "Cargo.lock", "src/circuit.rs", "src/main.rs", "src/statement.rs", "benchmark.py")
    report = {"platform": platform.system(), "python": platform.python_version(), "rayon_threads": 2,
              "verifier_sha256": sha(bin_path), "shape": shape, "setup": setup, "cohorts": rows,
              "sources_sha256": {p: sha(Path(__file__).with_name("Cargo.toml").parent / p) for p in sources}}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
