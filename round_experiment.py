#!/usr/bin/env python3
"""Run the synthetic LoRA round through separate contributor and verifier processes."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import oracle
import prover
import round_verifier as V
import tamper


HERE = Path(__file__).resolve().parent
SEEDS = (20261121, 90817)


def fixture(seed):
    rng = random.Random(seed)
    weights = [rng.randint(-2, 2) for _ in range(V.D * V.D)]
    a = [int(i % V.R == j) for j in range(V.R) for i in range(V.D)]
    initial = {"A": a, "B": [0] * (V.D * V.R)}
    teacher = {
        "A": a[:],
        "B": [rng.choice((-1, 1)) if i % V.R == j else 0
              for i in range(V.D) for j in range(V.R)],
    }
    for i in range(8, 16):
        teacher["A"][(i % V.R)*V.D+i] += rng.choice((-1, 1))

    def example(points):
        sample = [0] * V.D
        for i, value in points:
            sample[i] = value
        return sample

    def batch(rows):
        features = [v for row in rows for v in row]
        n = len(rows)
        base = oracle.gemm_rank1(rows, oracle.rows_of(weights, (V.D, V.D)), False, False, n, V.D, V.D)
        projected = oracle.gemm_rank1(rows, oracle.rows_of(teacher["A"], (V.R, V.D)), False, True, n, V.R, V.D)
        delta = oracle.gemm_rank1(projected, oracle.rows_of(teacher["B"], (V.D, V.R)), False, True, n, V.D, V.R)
        targets = [h + V.SCALE * v for base_row, delta_row in zip(base, delta, strict=True)
                   for h, v in zip(base_row, delta_row, strict=True)]
        return {"x": features, "target": targets}

    phases = []
    for phase, factor in enumerate(V.FACTORS):
        slots = []
        for i, name in enumerate(V.WORKERS):
            start = (phase * len(V.WORKERS) + i) * V.B
            points = [example([(k, 1)]) for k in range(start, start + V.B)]
            slots.append({"worker": name, **batch(points)})
        phases.append({"factor": factor, "slots": slots})
    heldout = [example([(8 + i % 8, 1 if i < 8 else -1), (16 + i, -1)])
               for i in range(V.EVAL_ROWS)]
    return {"schema": V.SCHEMA, "id": f"synthetic-{seed}", "W": weights, "initial": initial,
            "phases": phases, "heldout": batch(heldout), "max_heldout_sse": 0}, teacher


def proof_for(manifest, pin, phase, slot, adapter, lib=None, upstream=None):
    predicted = V.predict(manifest["W"], adapter, slot["x"])
    g = V.residual(predicted, slot["target"]) if upstream is None else upstream
    inputs = {"W": manifest["W"], **adapter, "x": slot["x"], "g": g}
    seed = V.slot_seed(pin, phase, slot["worker"])
    return prover.build_proof(V.D, V.R, V.B, V.SCALE, seed, "gpu" if lib else "cpu",
                              lib, reps=1, warmup=0, inputs=inputs)


def contribute(args):
    manifest, pin = V.read_json(args.manifest, V.MANIFEST_BYTES)
    V.validate_manifest(manifest)
    slot, = (s for s in manifest["phases"][args.phase]["slots"] if s["worker"] == args.worker)
    raw = sys.stdin.buffer.read(V.MANIFEST_BYTES + 1)
    if len(raw) > V.MANIFEST_BYTES:
        raise ValueError("adapter input exceeds size limit")
    adapter = json.loads(raw)
    lib = prover.load_lib(args.lib) if args.lib else None
    info = prover.gpu_info(lib) if lib else None
    if info and (info["compute_cap"] != "12.0" or "RTX 5060 Ti" not in info["gpu_name"]):
        raise RuntimeError("this trial requires the selected RTX 5060 Ti")
    proof, _ = proof_for(manifest, pin, args.phase, slot, adapter, lib)
    times = proof.pop("prover_timing")
    proof.pop("gemm_paths")
    proof.pop("timing_boundary")
    print(json.dumps({"proof": proof, "backend": info or "cpu", "timing": times}))


def assemble(manifest_path, path, lib):
    manifest, pin = V.read_json(manifest_path, V.MANIFEST_BYTES)
    V.validate_manifest(manifest)
    adapter = manifest["initial"]
    bundle = {"schema": V.SCHEMA, "manifest_sha256": pin, "phases": []}
    measurements = []
    start = time.perf_counter()
    for phase, plan in enumerate(manifest["phases"]):
        records = []
        for slot in plan["slots"]:
            worker = slot["worker"]
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "contribute",
                   str(manifest_path), str(phase), worker]
            if lib and worker == V.WORKERS[1]:
                cmd += ["--lib", str(lib)]
            reply = subprocess.run(cmd, input=V.canonical(adapter), capture_output=True,
                                   timeout=30, check=True)
            step = json.loads(reply.stdout)
            records.append({"worker": worker, "proof": step.pop("proof")})
            measurements.append({"phase": phase, "worker": worker, **step})
        next_adapter = V.advance(manifest, pin, phase, adapter, records)
        bundle["phases"].append({"previous": V.adapter_digest(pin, adapter),
                                 "contributions": records, "next": V.adapter_digest(pin, next_adapter)})
        adapter = next_adapter
    bundle.update(merkle_root=V.merkle_root(bundle["phases"]), final_adapter=V.adapter_digest(pin, adapter))
    V.save_json(path, bundle)
    build_wall = time.perf_counter() - start
    return bundle, {"build_wall_seconds": build_wall, "bundle_bytes": path.stat().st_size,
                    "contributions": measurements}


def controls(manifest, pin, bundle):
    second = manifest["phases"][1]
    prior = manifest["initial"]
    stale, expected = proof_for(manifest, pin, 1, second["slots"][0], prior)
    V.verify_step(stale, expected)
    tampered, _ = tamper.wrong_dW_coherent(bundle["phases"][0]["contributions"][0]["proof"])
    next_adapter = V.advance(manifest, pin, 0, prior, bundle["phases"][0]["contributions"])
    slot = second["slots"][0]
    wrong_gradient, expected = proof_for(manifest, pin, 1, slot, next_adapter, upstream=[0] * (V.D * V.B))
    V.verify_step(wrong_gradient, expected)

    def shorten(candidate):
        candidate["phases"][1]["contributions"].pop()

    def duplicate(candidate):
        records = candidate["phases"][0]["contributions"]
        records[1] = records[0]

    def relabel(candidate):
        records = candidate["phases"][0]["contributions"]
        records[1]["proof"] = records[0]["proof"]

    changes = (
        ("changed_optimizer_transition", lambda c: c["phases"][0].update(next="0"*64)),
        ("missing_phase", lambda c: c["phases"].pop()),
        ("missing_contribution", shorten),
        ("duplicate_contribution", duplicate),
        ("relabelled_contribution", relabel),
        ("valid_step_on_stale_adapter", lambda c: c["phases"][1]["contributions"][0].update(proof=stale)),
        ("valid_step_with_wrong_gradient", lambda c: c["phases"][1]["contributions"][0].update(proof=wrong_gradient)),
        ("resealed_wrong_matrix", lambda c: c["phases"][0]["contributions"][0].update(proof=tampered)),
        ("changed_round_root", lambda c: c.update(merkle_root="0"*64)),
        ("changed_final_digest", lambda c: c.update(final_adapter="0"*64)),
    )
    outcomes = {}
    for name, change in changes:
        altered = copy.deepcopy(bundle)
        change(altered)
        if name not in ("changed_round_root", "missing_phase", "missing_contribution"):
            altered["merkle_root"] = V.merkle_root(altered["phases"])
        try:
            V.verify_round(manifest, pin, altered)
        except V.Reject as exc:
            outcomes[name] = str(exc)
        else:
            raise RuntimeError("accepted negative control: " + name)
    return outcomes


def verify_cli(manifest, pin, bundle, adapter):
    cmd = [sys.executable, "-B", str(HERE / "round_verifier.py"), str(manifest), str(bundle),
           "--manifest-sha256", pin, "--adapter", str(adapter)]
    result = subprocess.run(cmd, capture_output=True, timeout=30, check=True)
    return json.loads(result.stdout)


def run(args):
    out = args.output
    out.mkdir(mode=0o700)
    selections = []
    for seed in SEEDS:
        home = out / str(seed)
        home.mkdir()
        manifest, teacher = fixture(seed)
        path = home / "manifest.json"
        V.save_json(path, manifest)
        selections.append((home, path, teacher))
    cohorts = []
    for home, path, teacher in selections:
        manifest, pin = V.read_json(path, V.MANIFEST_BYTES)
        cpu, cpu_metrics = assemble(path, home / "cpu.json", None)
        adapter = home / "adapter.json"
        saved = home / ("mixed.json" if args.lib else "cpu.json")
        mixed, mixed_metrics = assemble(path, saved, args.lib) if args.lib else (cpu, None)
        if cpu != mixed:
            raise RuntimeError("CPU replay differs from the mixed contributions")
        result = verify_cli(path, pin, saved, adapter)
        if json.loads(adapter.read_bytes()) != teacher:
            raise RuntimeError("the verifier did not recover the expected adapter")
        cohorts.append({"id": manifest["id"], "manifest_sha256": pin, "manifest_bytes": path.stat().st_size,
                        "cpu": cpu_metrics, "mixed": mixed_metrics, "replay": result,
                        "adapter_bytes": adapter.stat().st_size, "negative_controls": controls(manifest, pin, mixed)})
        print(f"{manifest['id']} SSE {result['initial_sse']} -> {result['final_sse']}, root {result['merkle_root']}")
    sources = ("round_verifier.py", "round_experiment.py", "prover.py", "contract.py", "verifier.py",
               "tamper.py", "oracle.py", "cuda/lora_kernels.cu")
    report = {"schema": V.SCHEMA, "python": platform.python_version(), "cohorts": cohorts,
              "source_sha256": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in sources}}
    if args.lib:
        report["cuda_library_sha256"] = hashlib.sha256(args.lib.read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    driver = commands.add_parser("run")
    driver.add_argument("--lib", type=lambda p: Path(p).resolve())
    driver.add_argument("--output", required=True, type=lambda p: Path(p).resolve())
    worker = commands.add_parser("contribute")
    worker.add_argument("manifest")
    worker.add_argument("phase", type=int, choices=range(len(V.FACTORS)))
    worker.add_argument("worker", choices=V.WORKERS)
    worker.add_argument("--lib")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        contribute(args)


if __name__ == "__main__":
    main()
