#!/usr/bin/env python3
"""Run two private Qwen orders against the same disclosed v2 circuit template."""

import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from benchmark import ROOT, sha, timed

sys.path.insert(0, str(ROOT))
import round_verifier as V


SOURCE_PIN = "ccb4d88429a4e2fce34b1f1a3d7cd88f1755f3ab6aaf9d91262ad0a52eb54263"
COMMAND = "private-prefix-"


def signed_columns(values, signs):
    return [value * signs[i % V.D] for i, value in enumerate(values)]


def private_fixture(source, recovered, column, sign):
    signs = [1 if secrets.randbits(1) else -1 for _ in range(V.D)]
    signs[column] = sign
    order = copy.deepcopy(source)
    order["initial"]["A"] = signed_columns(order["initial"]["A"], signs)
    order["W"] = [w * signs[i // V.D] for i, w in enumerate(order["W"])]
    for phase in order["phases"]:
        for slot in phase["slots"]:
            slot["x"] = signed_columns(slot["x"], signs)
    order["heldout"]["x"] = signed_columns(order["heldout"]["x"], signs)
    adapter = {**recovered, "A": signed_columns(recovered["A"], signs)}
    return order, adapter


def failing_accuracy(order, adapter):
    target = V.top1(order["heldout"]["target"])
    altered = copy.deepcopy(order)
    altered["max_heldout_sse"] = 10**12
    features = altered["heldout"]["x"]
    for i, wanted in enumerate(target):
        span = slice(i * V.D, (i + 1) * V.D)
        previous = features[span]
        opposite = [-v for v in previous]
        if V.top1(V.predict(order["W"], adapter, opposite)) == [wanted]:
            continue
        features[span] = opposite
        try:
            V.validate_manifest(altered)
        except V.Reject:
            features[span] = previous
            continue
        predictions = V.predict(order["W"], adapter, features)
        assert V.squared_error(predictions, altered["heldout"]["target"]) <= altered["max_heldout_sse"]
        assert V.top1(predictions) != target
        return altered
    raise RuntimeError("could not create a distinct evaluation row that fails correctness")


def run(executable, out, image):
    os.umask(0o077)
    os.environ["RAYON_NUM_THREADS"] = "2"
    out.mkdir(mode=0o700)
    public = out / "public"
    public.mkdir(mode=0o700)
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                                      text=True).strip()
    fixture = ROOT / "results" / "text-head-origin" / "cpu"
    template = ROOT / "results" / "compact-prefix" / "cpu" / "shape.json"
    source, source_pin = V.read_json(fixture / "manifest.json", V.MANIFEST_BYTES)
    if source_pin != SOURCE_PIN:
        raise V.Reject("reference CPU manifest changed")
    bundle, _ = V.read_json(fixture / "bundle.json", V.BUNDLE_BYTES)
    reference, _ = V.verify_round(source, source_pin, bundle)
    active_entry = next(i for i, value in enumerate(reference["A"]) if value)
    coordinate = active_entry % V.D

    def call(label, command, *args, reject=None):
        return timed(out, label, [executable, COMMAND + command, *args], reject=reject)

    shape = call("shape", "shape", template)
    with TemporaryDirectory(prefix="prefix-witness-", dir=out) as scratch:
        work = Path(scratch)
        keys = work / "keys"
        setup = call("setup", "setup", template, keys)
        vk = public / "verifying.key"
        shutil.copyfile(keys / vk.name, vk)
        vk_pin = sha(vk)
        published = []
        timings = []
        for ordinal, sign in enumerate((-1, 1), start=1):
            label = f"order-{ordinal}"
            folder = work / label
            folder.mkdir(mode=0o700)
            order, adapter = private_fixture(source, reference, coordinate, sign)
            manifest_path, adapter_path = folder / "manifest.json", folder / "adapter.json"
            V.save_json(manifest_path, order)
            V.save_json(adapter_path, adapter)
            pin = sha(manifest_path)
            customer, worker = folder / "customer", folder / "worker"
            issued = call(label + "-issue", "order", template, manifest_path, pin, customer)
            order_pin = sha(customer / "order.commitment")
            assert order_pin == issued["reported"]["metrics"]["order_commitment_sha256"]
            sealed = call(label + "-seal", "output", customer / "order.commitment", adapter_path, worker)
            assert all(not p.stat().st_mode & 0o077 for p in (customer, worker))

            public_order = public / label
            public_order.mkdir(mode=0o700)
            packet, proof = public_order / "round.commitments", public_order / "round.proof"
            shutil.copyfile(worker / packet.name, packet)
            order_salt, output_salt = customer / "order.salt", worker / "output.salt"
            assert packet.read_bytes()[:32] == (customer / "order.commitment").read_bytes()
            produced = call(label + "-prove", "prove", template, manifest_path, pin, adapter_path,
                            order_salt, output_salt, packet, keys / "proving.key", proof)

            container = [
                "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
                "--cap-drop=all", "--security-opt=no-new-privileges", "--pids-limit=64",
                "--memory=192m", "--cpus=1", "--user", f"{os.getuid()}:{os.getgid()}",
                "--mount", f"type=bind,src={packet},dst=/round.commitments,readonly",
                "--mount", f"type=bind,src={proof},dst=/round.proof,readonly",
                "--mount", f"type=bind,src={vk},dst=/verifying.key,readonly",
                "--mount", f"type=bind,src={executable},dst=/verify-round,readonly",
                image_id, "/verify-round", COMMAND + "verify", "/round.commitments", order_pin,
                "/verifying.key", vk_pin, "/round.proof",
            ]
            arbiter = timed(out, label + "-arbiter", container)
            opened = call(label + "-open", "open-output", adapter_path, output_salt, packet, order_pin)
            timings.append({
                "issue": issued, "seal": sealed, "prove": produced, "arbiter": arbiter,
                "open": opened, "order_commitment_sha256": order_pin,
            })
            published.append({
                "manifest": manifest_path, "adapter": adapter_path, "order": order, "result": adapter,
                "packet": packet, "proof": proof, "pin": order_pin, "customer": customer,
                "order_salt": order_salt, "output_salt": output_salt,
            })

        first, second = published
        assert first["manifest"].read_bytes() != second["manifest"].read_bytes()
        assert first["adapter"].read_bytes() != second["adapter"].read_bytes()
        assert first["packet"].read_bytes() != second["packet"].read_bytes()
        expected_files = {"verifying.key", "order-1/round.commitments", "order-1/round.proof",
                          "order-2/round.commitments", "order-2/round.proof"}
        actual_files = {path.relative_to(public).as_posix()
                        for path in public.rglob("*") if path.is_file()}
        assert actual_files == expected_files

        controls = {}
        controls["cross_order_proof"] = call("cross-order", "verify", second["packet"], second["pin"],
                                              vk, vk_pin, first["proof"], reject="Groth16 rejected")
        mixed = work / "mixed.commitments"
        mixed.write_bytes(first["packet"].read_bytes()[:32] + second["packet"].read_bytes()[32:])
        controls["cross_order_output"] = call("cross-output", "verify", mixed, first["pin"], vk,
                                               vk_pin, first["proof"], reject="Groth16 rejected")
        wrong = work / "changed-adapter.json"
        adapter = copy.deepcopy(first["result"])
        adapter["A"][active_entry] = 0
        V.save_json(wrong, adapter)
        controls["changed_adapter_opening"] = call("changed-output", "open-output", wrong,
                                                    first["output_salt"], first["packet"], first["pin"],
                                                    reject="private prefix adapter opening mismatch")

        lower = work / "lower-ceiling.json"
        order = {**first["order"], "max_heldout_sse": 0}
        V.save_json(lower, order)
        strict = work / "strict-order"
        rejected = work / "strict.proof"
        controls["changed_ceiling_opening"] = call(
            "changed-ceiling", "prove", template, lower, sha(lower), first["adapter"],
            first["order_salt"], first["output_salt"], first["packet"], keys / "proving.key",
            rejected, reject="private prefix opening does not match",
        )
        call("strict-issue", "order", template, lower, sha(lower), strict)
        strict_output = work / "strict-output"
        call("strict-seal", "output", strict / "order.commitment", first["adapter"], strict_output)
        controls["fresh_order_failing_error_gate"] = call(
            "strict-proof", "prove", template, lower, sha(lower), first["adapter"],
            strict / "order.salt", strict_output / "output.salt", strict_output / "round.commitments",
            keys / "proving.key", rejected, reject="private prefix circuit rejected the release statement",
        )
        assert not rejected.exists()

        bad_accuracy = work / "wrong-accuracy.json"
        V.save_json(bad_accuracy, failing_accuracy(first["order"], first["result"]))
        bad_pin = sha(bad_accuracy)
        bad_order, bad_result = work / "wrong-order", work / "wrong-result"
        call("accuracy-issue", "order", template, bad_accuracy, bad_pin, bad_order)
        call("accuracy-seal", "output", bad_order / "order.commitment", first["adapter"], bad_result)
        controls["fresh_order_failing_accuracy_gate"] = call(
            "accuracy-proof", "prove", template, bad_accuracy, bad_pin, first["adapter"],
            bad_order / "order.salt", bad_result / "output.salt", bad_result / "round.commitments",
            keys / "proving.key", rejected, reject="private prefix circuit rejected the release statement",
        )
        assert not rejected.exists()

        alternate = work / "altered-template.json"
        variant = json.loads(template.read_text())
        variant["orders"][1][0], variant["orders"][1][1] = variant["orders"][1][1], variant["orders"][1][0]
        V.save_json(alternate, variant)
        controls["changed_template"] = call(
            "changed-template", "prove", alternate, first["manifest"], sha(first["manifest"]),
            first["adapter"], first["order_salt"], first["output_salt"], first["packet"],
            keys / "proving.key", rejected, reject="private prefix opening does not match",
        )

        fresh = work / "fresh-order"
        call("fresh-salt", "order", template, first["manifest"], sha(first["manifest"]), fresh)
        assert sha(fresh / "order.commitment") != first["pin"]

    report = {
        "scope": "private numeric Qwen v2 order under a disclosed circuit template",
        "template_sha256": sha(template), "container_image_id": image_id,
        "verifier_sha256": sha(executable), "shape": shape, "setup": setup,
        "orders": timings, "controls": controls, "reuse_one_verification_key": True,
        "same_order_fresh_salt_differed": True,
        "public_files": {path.relative_to(public).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
                         for path in sorted(public.rglob("*")) if path.is_file()},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Two private prefix orders verified under one public template and one key.")
    print(f"The customer opened both adapters and {len(controls)} controls were rejected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=lambda path: Path(path).resolve())
    parser.add_argument("output", type=lambda path: Path(path).resolve())
    parser.add_argument("--image", default="ubuntu:24.04")
    args = parser.parse_args()
    run(args.executable, args.output, args.image)
