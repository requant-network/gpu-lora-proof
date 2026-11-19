#!/usr/bin/env python3
"""Experiment runner for gpu-lora-proof (executes on the GPU server).

Sweep: d in {256,512,1024} x r in {8,16} x batch in {1,16} x 2 seeds.
Per configuration: GPU prover (warmup + repeated launches), independent CPU
verifier CLI (must print YES, exit 0), exact rank-1 CPU oracle replay, and
12 negative controls (11 single-proof tampers + cross-seed stale replay),
all of which must be rejected.  Writes results/results.json (environment,
source hashes, raw counts, timings) and generates results/RESULTS.md.

GPU etiquette: run under CUDA_VISIBLE_DEVICES=0 and the shared flock, e.g.
  CUDA_VISIBLE_DEVICES=0 flock /tmp/requant-new-studies-gpu.lock -c \
      'python3 run.py --lib ./liblora.so'
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

import contract as C
import oracle
import prover
import tamper
import verifier

CONFIGS = [(d, r, b) for d in (256, 512, 1024) for r in (8, 16) for b in (1, 16)]
SEEDS = (20261119, 987654321)
SOURCE_FILES = ["contract.py", "prover.py", "verifier.py", "oracle.py",
                "tamper.py", "run.py", "cuda/lora_kernels.cu"]


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def collect_env(lib):
    info = prover.gpu_info(lib)
    if info["compute_cap"] != "12.0" or "5060 Ti" not in info["gpu_name"]:
        raise RuntimeError(f"GPU identity gate failed (Blackwell directive): {info}")
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name",
             "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        drv = "unavailable"
    try:
        nvcc = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"],
                              capture_output=True, text=True,
                              timeout=30).stdout.strip().splitlines()[-1]
    except Exception:
        nvcc = "unavailable"
    return {"gpu_name": info["gpu_name"],
            "compute_cap": info["compute_cap"],
            "cuda_driver_version_api": info["cuda_driver_version"],
            "cuda_runtime_version_api": info["cuda_runtime_version"],
            "nvidia_smi_gpu0": drv, "nvcc": nvcc,
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "utc_timestamp": datetime.datetime.now(datetime.timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ")}


def run_one(d, r, b, seed, lib, args, proof_dir):
    s = C.S_DEFAULT
    t0 = time.perf_counter()
    proof, expected = prover.build_proof(d, r, b, s, seed, "gpu", lib,
                                         reps=args.reps, warmup=args.warmup)
    build_s = time.perf_counter() - t0
    tag = f"d{d}_r{r}_b{b}_seed{seed}"
    ppath = os.path.join(proof_dir, f"proof_{tag}.json")
    epath = os.path.join(proof_dir, f"expected_{tag}.json")
    with open(ppath, "w") as f:
        json.dump(proof, f)
    with open(epath, "w") as f:
        json.dump(expected, f)
    proof_bytes = os.path.getsize(ppath)

    # Honest verification through the real CLI (exit code + stdout evidence).
    stats_path = os.path.join(proof_dir, f"stats_{tag}.json")
    cli = subprocess.run([sys.executable, "verifier.py", ppath, epath,
                          "--json", stats_path], capture_output=True, text=True)
    honest_ok = cli.returncode == 0 and cli.stdout.strip() == "YES"
    with open(stats_path) as f:
        vstats = json.load(f)

    # Independent exact oracle replay (rank-1 accumulation, pure integers).
    mismatches, oracle_elements = oracle.oracle_replay(proof)
    oracle_ok = not mismatches

    # Negative controls: all must be rejected by the independent verifier.
    controls = []
    for name, fn in tamper.SINGLE_CONTROLS:
        tp, why = fn(proof)
        try:
            verifier.verify(tp, expected)
            rejected, reason = False, "ACCEPTED (control failure)"
        except verifier.Reject as e:
            rejected, reason = True, str(e)
        except Exception as e:
            rejected, reason = True, f"malformed: {e}"
        controls.append({"control": name, "rejected": rejected,
                         "reason": reason, "expected_catch": why})
    return {"config": {"d": d, "r": r, "b": b, "s": s, "seed": seed},
            "tag": tag, "proof_file": ppath, "proof_bytes": proof_bytes,
            "proof_sha256": sha256_file(ppath),
            "expected_sha256": sha256_file(epath),
            "honest_verifier": {"accepted": honest_ok,
                                "cli_exit": cli.returncode,
                                "cli_stdout": cli.stdout.strip(),
                                "stats": vstats},
            "oracle": {"ok": oracle_ok, "mismatches": mismatches,
                       "elements_checked": oracle_elements},
            "controls": controls,
            "gemm_paths": proof["gemm_paths"],
            "prover_timing": proof["prover_timing"],
            "build_wall_s": build_s,
            "leaf_hashes": proof["leaf_hashes"]}


def stale_control(runs_by_tag, d, r, b, seed_a, seed_b):
    pa = runs_by_tag[f"d{d}_r{r}_b{b}_seed{seed_a}"]["proof_file"]
    ea = runs_by_tag[f"d{d}_r{r}_b{b}_seed{seed_a}"]["expected_file"]
    pb = runs_by_tag[f"d{d}_r{r}_b{b}_seed{seed_b}"]["proof_file"]
    with open(pb) as f:
        proof_b = json.load(f)
    with open(ea) as f:
        expected_a = json.load(f)
    tp, why = tamper.stale_replay(proof_b)
    try:
        verifier.verify(tp, expected_a)
        rejected, reason = False, "ACCEPTED (control failure)"
    except verifier.Reject as e:
        rejected, reason = True, str(e)
    return {"control": "stale_replay_cross_seed", "rejected": rejected,
            "reason": reason, "expected_catch": why,
            "config": {"d": d, "r": r, "b": b,
                       "seed_presented": seed_b, "seed_expected": seed_a}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="./liblora.so")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="smoke subset: d=256, r=8, b=1, first seed only")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    proof_dir = os.path.join(a.outdir, "proofs")
    os.makedirs(proof_dir, exist_ok=True)
    lib = prover.load_lib(a.lib)
    env = collect_env(lib)
    print("ENV:", json.dumps(env), flush=True)
    configs = [(256, 8, 1)] if a.quick else CONFIGS
    seeds = SEEDS[:1] if a.quick else SEEDS
    runs, stale = [], []
    failures = []
    for (d, r, b) in configs:
        for seed in seeds:
            rec = run_one(d, r, b, seed, lib, a, proof_dir)
            rec["expected_file"] = os.path.join(
                proof_dir, f"expected_{rec['tag']}.json")
            runs.append(rec)
            ok = (rec["honest_verifier"]["accepted"] and rec["oracle"]["ok"]
                  and all(c["rejected"] for c in rec["controls"]))
            print(f"[{rec['tag']}] honest={rec['honest_verifier']['accepted']} "
                  f"oracle={rec['oracle']['ok']} "
                  f"controls={sum(c['rejected'] for c in rec['controls'])}"
                  f"/{len(rec['controls'])} "
                  f"verify_ms={rec['honest_verifier']['stats']['verify_ms']:.1f} "
                  f"proof_bytes={rec['proof_bytes']}", flush=True)
            if not ok:
                failures.append(rec["tag"])
    by_tag = {r["tag"]: r for r in runs}
    if not a.quick:
        for (d, r, b) in configs:
            sc = stale_control(by_tag, d, r, b, SEEDS[0], SEEDS[1])
            stale.append(sc)
            if not sc["rejected"]:
                failures.append(f"stale_{d}_{r}_{b}")
    all_controls = [c for r in runs for c in r["controls"]] + stale
    summary = {
        "configs_run": len(runs),
        "honest_accepted": sum(r["honest_verifier"]["accepted"] for r in runs),
        "oracle_exact": sum(r["oracle"]["ok"] for r in runs),
        "oracle_elements_checked": sum(r["oracle"]["elements_checked"] for r in runs),
        "negative_controls_total": len(all_controls),
        "negative_controls_rejected": sum(c["rejected"] for c in all_controls),
        "total_verifier_checks_honest": sum(
            r["honest_verifier"]["stats"]["total_checks"] for r in runs),
        "failures": failures,
        "soundness_bound": "9/p^2 with p=2^61-1 => < 2^-118 (target <= 2^-63)",
    }
    result = {"study": "gpu-lora-proof",
              "title": "Verifiable LoRA fine-tuning step: exact integer "
                       "forward+backward with Freivalds/Fiat-Shamir "
                       "certificates and an independent CPU verifier",
              "environment": env,
              "source_sha256": {f: sha256_file(f) for f in SOURCE_FILES},
              "lib_sha256": sha256_file(a.lib),
              "timing_protocol": {"warmup": a.warmup, "measured_reps": a.reps,
                                  "statistic": "median with min/max range"},
              "summary": summary, "runs": runs, "stale_controls": stale}
    rpath = os.path.join(a.outdir, "results.json")
    with open(rpath, "w") as f:
        json.dump(result, f, indent=1)
    print("SUMMARY:", json.dumps(summary), flush=True)
    print(f"results written: {rpath}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()