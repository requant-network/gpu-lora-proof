"""Compare deterministic replay, a transparent certificate and Groth16 for public Qwen v2 features."""
import argparse
import gzip
import hmac
import io
import json
import os
from pathlib import Path
from statistics import median
import sys
import tarfile
from time import perf_counter

from benchmark import ROOT, sha, timed

sys.path.insert(0, str(ROOT))
import round_verifier as V
from prover import cpu_step


COHORTS = {
    "rtx-features": ("results/text-head-v2/mixed-vocabulary", "results/compact-prefix",
                     "d200e82458d79b44fc577480ba33b1e7fe79b6e935328bd7812cfdd05763c34a"),
    "cpu-features": ("results/text-head-origin/cpu", "results/compact-prefix/cpu",
                     "ccb4d88429a4e2fce34b1f1a3d7cd88f1755f3ab6aaf9d91262ad0a52eb54263"),
}


def replay(manifest_path, pin, adapter_path):
    manifest, actual = V.read_json(manifest_path, V.MANIFEST_BYTES)
    if not hmac.compare_digest(actual, pin):
        raise V.Reject("manifest SHA-256 mismatch")
    V.validate_manifest(manifest)
    if manifest["schema"] != V.PREFIX_SCHEMA or manifest.get("min_heldout_correct") != V.EVAL_ROWS:
        raise V.Reject("expected the supported full-accuracy v2 release gate")
    proposed, _ = V.read_json(adapter_path, 8 << 10)
    if (set(proposed) != {"A", "B"} or not V.bounded_ints(proposed["A"], V.R * V.D, V.C.A_MAX)
            or not V.bounded_ints(proposed["B"], V.D * V.R, V.C.B_MAX)):
        raise V.Reject("invalid claimed adapter")
    adapter = manifest["initial"]
    weights = manifest["W"]
    for phase in manifest["phases"]:
        factor = phase["factor"]
        gradients = []
        for slot in phase["slots"]:
            gradient = V.residual(V.predict(weights, adapter, slot["x"]), slot["target"])
            arrays = {"W": weights, "x": slot["x"], "g": gradient, **adapter}
            gradients.append(cpu_step(V.D, V.R, V.B, V.SCALE, arrays)["d" + factor])
        totals = [sum(pair) for pair in zip(*gradients, strict=True)]
        updated = V.sign_step(adapter[factor], factor, totals)
        adapter = V.choose_prefix(weights, phase["slots"], adapter, factor, updated, totals)
    if adapter != proposed:
        raise V.Reject("claimed adapter differs from direct replay")
    heldout = manifest["heldout"]
    prediction = V.predict(weights, adapter, heldout["x"])
    sse = V.squared_error(prediction, heldout["target"])
    if sse > manifest["max_heldout_sse"] or V.top1(prediction) != V.top1(heldout["target"]):
        raise V.Reject("held-out release gate failed")
    return {"final_adapter": V.adapter_digest(pin, adapter, V.PREFIX_SCHEMA),
            "final_sse": sse, "heldout_correct": V.EVAL_ROWS}


def package(path, files):
    with path.open("xb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for name, source in sorted(files.items()):
                data = source.read_bytes()
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    return {"raw_bytes": sum(file.stat().st_size for file in files.values()),
            "gzip_bytes": path.stat().st_size, "sha256": sha(path)}


def receive(path, target, names):
    target.mkdir()
    with tarfile.open(path, mode="r:gz") as archive:
        for name in sorted(names):
            entry = archive.next()
            if entry is None or entry.name != name or not entry.isfile() or not 0 < entry.size <= 1 << 20:
                raise ValueError("unexpected package member")
            (target / name).write_bytes(archive.extractfile(entry).read())
        if archive.next() is not None:
            raise ValueError("extra package member")


def trial(root, executable, fixture, proof_fixture, pin):
    manifest, adapter, bundle = (fixture / n for n in ("manifest.json", "adapter.json", "bundle.json"))
    shape = proof_fixture / "shape.json"
    keys = root / "keys"
    setup = timed(root, "setup", [executable, "prefix-setup", shape, keys])
    key = keys / "verifying.key"
    key_pin = sha(key)
    proof = root / "fresh.proof"
    prover = timed(root, "fresh-proof", [executable, "prefix-prove", shape, manifest, pin, adapter,
                                         keys / "proving.key", proof])
    if proof.stat().st_size != 128:
        raise ValueError("unexpected public proof size")
    saved = root / "rebuilt-bundle.json"
    assemble = ("import json,sys;from pathlib import Path;from round_experiment import assemble;"
                "_,r=assemble(Path(sys.argv[1]),Path(sys.argv[2]),None);print(json.dumps(r))")
    builder = timed(root, "assemble", [sys.executable, "-B", "-c", assemble, manifest, saved])
    if saved.read_bytes() != bundle.read_bytes():
        raise ValueError("transparent assembly changed the retained certificate")
    direct = {"manifest.json": manifest, "adapter.json": adapter}
    proof_pack = {**direct, "round.proof": proof}
    inputs = {"direct": direct, "transparent": {**direct, "bundle.json": bundle},
              "cached_proof": proof_pack, "first_proof": {**proof_pack, "verifying.key": key}}
    packets = {name: package(root / f"{name}.tar.gz", files) for name, files in inputs.items()}
    measures = {name: [] for name in inputs}
    methods = list(inputs)
    script = [sys.executable, "-B", Path(__file__).resolve(), "replay"]
    for repeat in range(4):
        for name in methods[repeat:] + methods[:repeat]:
            label = f"{name}-{repeat}"
            target = root / label
            started = perf_counter()
            receive(root / f"{name}.tar.gz", target, inputs[name])
            unpack_seconds = perf_counter() - started
            statement, claim = target / "manifest.json", target / "adapter.json"
            if name == "direct":
                command = [*script, statement, pin, claim]
            elif name == "transparent":
                command = [sys.executable, "-B", ROOT / "round_verifier.py", statement, target / "bundle.json",
                           "--manifest-sha256", pin, "--adapter", target / "recovered.json"]
            else:
                vk = target / "verifying.key" if name == "first_proof" else key
                command = [executable, "prefix-verify", statement, pin, claim, vk, key_pin, target / "round.proof"]
            measured = timed(root, label, command)
            if name == "transparent" and (target / "recovered.json").read_bytes() != claim.read_bytes():
                raise ValueError("transparent replay disagrees with the claimed adapter")
            measures[name].append({"unpack_seconds": unpack_seconds, **measured})
    claimed, _ = V.read_json(adapter, 8 << 10)
    claimed["A"][0] += -1 if claimed["A"][0] == V.C.A_MAX else 1
    changed = root / "changed.json"
    V.save_json(changed, claimed)
    direct_control = timed(root, "wrong-direct", [*script, manifest, pin, changed],
                           reject="claimed adapter differs from direct replay")
    proof_control = timed(root, "wrong-proof", [executable, "prefix-verify", manifest, pin, changed, key, key_pin, proof],
                          reject="Groth16 rejected")
    result = replay(manifest, pin, adapter)
    for name in ("direct", "transparent"):
        for sample in measures[name]:
            row = sample["reported"]
            if row["final_adapter"] != result["final_adapter"] or row["final_sse"] != result["final_sse"]:
                raise ValueError("the two replay paths calculated different releases")
    return {"manifest_sha256": pin, "adapter_sha256": sha(adapter), "verification_key_sha256": key_pin,
            "setup": setup, "prover": prover, "transparent_assembly": builder, "packets": packets,
            "receiver_trials": measures, "direct_result": result,
            "receiver_median_seconds": {name: median(m["unpack_seconds"] + m["process_seconds"] for m in rows)
                                         for name, rows in measures.items()},
            "controls": {"direct": direct_control, "proof": proof_control}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    direct = subs.add_parser("replay")
    direct.add_argument("manifest", type=Path)
    direct.add_argument("pin")
    direct.add_argument("adapter", type=Path)
    bench = subs.add_parser("benchmark")
    bench.add_argument("executable", type=lambda p: Path(p).resolve())
    bench.add_argument("output", type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    try:
        if args.mode == "replay":
            print(json.dumps(replay(args.manifest, args.pin, args.adapter)))
        else:
            os.environ["RAYON_NUM_THREADS"] = "2"
            args.output.mkdir(mode=0o700)
            results = {}
            for name, (fixture, proof_fixture, pin) in COHORTS.items():
                target = args.output / name
                target.mkdir()
                results[name] = trial(target, args.executable, ROOT / fixture, ROOT / proof_fixture, pin)
            (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    except (V.Reject, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"NO: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
