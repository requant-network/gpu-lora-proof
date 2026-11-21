#!/usr/bin/env python3
"""Exercise a private order, its linked adapter and a public-only arbiter."""

import argparse
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
from round_experiment import fixture
from round_verifier import save_json


def run(executable, out, image):
    os.umask(0o077)
    os.environ["RAYON_NUM_THREADS"] = "2"
    out.mkdir(mode=0o700)
    public = out / "public"
    public.mkdir(mode=0o700)
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    shape = timed(out, "shape", [executable, "private-shape"])
    with TemporaryDirectory(prefix="witness-", dir=out) as scratch:
        work = Path(scratch)
        manifest, adapter = work / "manifest.json", work / "adapter.json"
        order, result = fixture(secrets.randbits(128))
        save_json(manifest, order)
        save_json(adapter, result)
        pin = sha(manifest)
        keys, customer, worker = (work / folder for folder in ("keys", "customer", "worker"))
        setup = timed(out, "setup", [executable, "private-setup", keys])
        order_seal = timed(out, "client-order", [executable, "private-order", manifest, pin, customer])
        assert not customer.stat().st_mode & 0o077
        order_pin = sha(customer / "order.commitment")
        assert order_pin == order_seal["reported"]["metrics"]["order_commitment_sha256"]
        result_seal = timed(out, "worker-result", [executable, "private-output", customer / "order.commitment", adapter, worker])
        assert not worker.stat().st_mode & 0o077
        order_salt, output_salt = customer / "order.salt", worker / "output.salt"
        packet = worker / "round.commitments"
        assert packet.read_bytes()[:32] == (customer / "order.commitment").read_bytes()
        pk, vk, proof = keys / "proving.key", keys / "verifying.key", public / "round.proof"
        prove = timed(out, "worker-prove", [executable, "private-prove", manifest, pin, adapter,
                                           order_salt, output_salt, packet, pk, proof])
        for source in (packet, vk):
            shutil.copyfile(source, public / source.name)
        vk_pin = sha(vk)
        assert set(p.name for p in public.iterdir()) == {"round.commitments", "round.proof", "verifying.key"}
        assert len(packet.read_bytes()) == 64 and proof.stat().st_size == 128
        # The arbiter only mounts the three published files and the executable.
        container = [
            "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
            "--cap-drop=all", "--security-opt=no-new-privileges", "--pids-limit=64",
            "--memory=192m", "--cpus=1", "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=bind,src={public},dst=/evidence,readonly",
            "--mount", f"type=bind,src={executable},dst=/verify-round,readonly",
            image_id, "/verify-round", "private-verify", "/evidence/round.commitments",
            order_pin, "/evidence/verifying.key", vk_pin, "/evidence/round.proof",
        ]
        arbiter = timed(out, "public-verifier", container)
        opened = timed(out, "client-opening", [executable, "private-open-output", adapter, output_salt, packet, order_pin])

        rerandomized = work / "second-order"
        timed(out, "client-fresh-salt", [executable, "private-order", manifest, pin, rerandomized])
        other_order = (rerandomized / "order.commitment").read_bytes()
        other_pin = sha(rerandomized / "order.commitment")
        assert other_order != packet.read_bytes()[:32]
        other_output = work / "second-output"
        timed(out, "worker-fresh-salt", [executable, "private-output", customer / "order.commitment", adapter, other_output])
        previous = packet.read_bytes()
        assert (other_output / "round.commitments").read_bytes()[:32] == previous[:32]
        assert (other_output / "round.commitments").read_bytes()[32:] != previous[32:]
        wrong_order, wrong_output = work / "wrong-order", other_output / "round.commitments"
        wrong_order.write_bytes(other_order + previous[32:])

        def verify_reject(label, altered, expected_order, reason):
            return timed(out, label, [executable, "private-verify", altered, expected_order, vk, vk_pin, proof],
                         reject=reason)

        groth = "Groth16 rejected the linked private commitments"
        controls = {
            "swapped_order_commitment": verify_reject("swapped-order", wrong_order, order_pin, "issued order commitment SHA-256 mismatch"),
            "old_proof_with_new_order_pin": verify_reject("relabelled-order", wrong_order, other_pin, groth),
            "swapped_output_commitment": verify_reject("swapped-output", wrong_output, order_pin, groth),
        }
        altered_manifest = work / "changed-weight.json"
        order["W"][0] += 1
        save_json(altered_manifest, order)
        rejected_proof = work / "rejected.proof"
        controls["changed_input_opening"] = timed(out, "changed-input", [
            executable, "private-prove", altered_manifest, sha(altered_manifest), adapter, order_salt,
            output_salt, packet, pk, rejected_proof,
        ], reject="private opening does not match")
        assert not rejected_proof.exists()
        changed = work / "changed-adapter.json"
        result["A"][0] += 1
        save_json(changed, result)
        controls["changed_adapter_opening"] = timed(out, "changed-adapter", [
            executable, "private-open-output", changed, output_salt, packet, order_pin,
        ], reject="private adapter opening mismatch")
        controls["wrong_key_pin"] = timed(out, "wrong-key", [
            executable, "private-verify", packet, order_pin, vk, "0"*64, proof,
        ], reject="verifying key SHA-256 mismatch")
        controls["cross_order_adapter_opening"] = timed(out, "cross-order-opening", [
            executable, "private-open-output", adapter, output_salt, wrong_order, other_pin,
        ], reject="private adapter opening mismatch")
        corrupt = work / "extra-proof"
        corrupt.write_bytes(proof.read_bytes() + b"\x00")
        controls["trailing_proof_bytes"] = timed(out, "extra-proof", [
            executable, "private-verify", packet, order_pin, vk, vk_pin, corrupt,
        ], reject="file exceeds size limit")

    paths = ("Cargo.toml", "Cargo.lock", "src/circuit.rs", "src/main.rs", "src/private.rs", "src/statement.rs",
             "private_trial.py", "benchmark.py")
    report = {"scope": "synthetic-lora-public-arbiter-v1", "container_image_id": image_id,
              "verifier_sha256": sha(executable), "shape": shape, "setup": setup,
              "client_commitment": order_seal, "worker_commitment": result_seal, "prove": prove,
              "arbiter": arbiter, "client_opening": opened, "controls": controls,
              "rerandomized_commitments_differed": True,
              "public_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(public.iterdir())},
              "sources_sha256": {p: sha(Path(__file__).resolve().parent / p) for p in paths}}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("private order proved; public-only offline arbiter accepted; client opened adapter")
    print(f"{len(controls)} controls rejected; fresh salts changed both commitments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=lambda path: Path(path).resolve())
    parser.add_argument("output", type=lambda path: Path(path).resolve())
    parser.add_argument("--image", default="ubuntu:24.04")
    args = parser.parse_args()
    run(args.executable, args.output, args.image)
