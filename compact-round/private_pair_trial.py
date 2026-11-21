#!/usr/bin/env python3
"""Prove two private prefix orders with one key and a hidden choice of template."""

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from tempfile import TemporaryDirectory

from benchmark import ROOT, sha, timed
from private_prefix_trial import COMMAND, SOURCE_PIN, V, private_fixture


def run(executable, out, image, compare_key_reuse=False):
    os.umask(0o077)
    os.environ["RAYON_NUM_THREADS"] = "2"
    out.mkdir()
    public = out / "public"
    public.mkdir()
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True,
    ).strip()
    results = ROOT / "results"
    profiles = (
        (results / "compact-prefix/cpu/shape.json", results / "text-head-origin/cpu", SOURCE_PIN),
        (results / "compact-prefix/shape.json", results / "text-head-v2/mixed-vocabulary",
         "d200e82458d79b44fc577480ba33b1e7fe79b6e935328bd7812cfdd05763c34a"),
    )
    assert sha(profiles[0][0]) != sha(profiles[1][0])
    fixtures = []
    references = []
    for shape, fixture, pin in profiles:
        manifest = fixture / "manifest.json"
        order, digest = V.read_json(manifest, V.MANIFEST_BYTES)
        if digest != pin:
            raise V.Reject("decoy manifest changed")
        bundle, _ = V.read_json(fixture / "bundle.json", V.BUNDLE_BYTES)
        adapter, _ = V.verify_round(order, pin, bundle)
        fixtures.extend([shape, manifest, pin, fixture / "adapter.json"])
        references.append((order, adapter))

    def call(label, command, *args, reject=None):
        return timed(out, label, [executable, COMMAND + command, *args], reject=reject)

    shape_result = call("shape", "pair-shape", *fixtures)
    captures = []
    controls = {}
    with TemporaryDirectory(prefix="pair-witness-", dir=out) as scratch:
        work = Path(scratch)
        keys = work / "keys"
        setup = call("setup", "pair-setup", keys, *fixtures)
        vk = public / "verifying.key"
        shutil.copyfile(keys / vk.name, vk)
        vk_pin = sha(vk)

        def issue_and_seal(tag, template, manifest, adapter, directory):
            customer, worker = directory / "customer", directory / "worker"
            issue = call(tag + "-issue", "order", template, manifest, sha(manifest), customer)
            seal = call(tag + "-seal", "output", customer / "order.commitment", adapter, worker)
            pin = sha(customer / "order.commitment")
            assert pin == issue["reported"]["metrics"]["order_commitment_sha256"]
            return customer / "order.salt", worker / "output.salt", worker / "round.commitments", pin, issue, seal

        def verify_proof(tag, commitments, proof, order_pin):
            verify = [
                "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
                "--cap-drop=all", "--security-opt=no-new-privileges", "--pids-limit=64",
                "--memory=192m", "--cpus=1", "--user", f"{os.getuid()}:{os.getgid()}",
                "--mount", f"type=bind,src={commitments},dst=/round.commitments,readonly",
                "--mount", f"type=bind,src={proof},dst=/round.proof,readonly",
                "--mount", f"type=bind,src={vk},dst=/verifying.key,readonly",
                "--mount", f"type=bind,src={executable},dst=/verify-round,readonly",
                image_id, "/verify-round", COMMAND + "verify", "/round.commitments", order_pin,
                "/verifying.key", vk_pin, "/round.proof",
            ]
            return timed(out, tag, verify)

        batch_orders = []
        shared = None
        choices = [0, 1]
        secrets.SystemRandom().shuffle(choices)
        for ordinal, choice in enumerate(choices, start=1):
            tag = f"order-{ordinal}"
            folder = work / tag
            folder.mkdir()
            order, original = references[choice]
            coordinate = next(i % V.D for i, value in enumerate(original["A"]) if value)
            private_order, adapter = private_fixture(order, original, coordinate, -1)
            manifest, adapter_path = folder / "manifest.json", folder / "adapter.json"
            V.save_json(manifest, private_order)
            V.save_json(adapter_path, adapter)
            assert sha(manifest) != profiles[choice][2]
            assert adapter != original
            route = issue_and_seal(tag, profiles[choice][0], manifest, adapter_path, folder)
            order_salt, output_salt, packet, order_pin, issue, seal = route
            published = public / tag
            published.mkdir()
            commitments, proof = published / packet.name, published / "round.proof"
            shutil.copyfile(packet, commitments)
            order_args = [manifest, sha(manifest), adapter_path, order_salt, output_salt, commitments]
            produced = call(tag + "-prove", "pair-prove", choice, *fixtures, *order_args, keys / "proving.key", proof)
            batch_orders.append(([choice, *order_args], commitments, order_pin, published))
            arbiter = verify_proof(tag + "-arbiter", commitments, proof, order_pin)
            opened = call(tag + "-open", "open-output", adapter_path, output_salt, commitments, order_pin)
            captures.append({
                "order_commitment_sha256": order_pin, "issue": issue, "seal": seal,
                "prove": produced, "arbiter": arbiter, "open": opened,
            })
            if ordinal == 1:
                refused = work / "refused.proof"
                other = 1 - choice
                controls["route_switch"] = call(
                    "route-switch", "pair-prove", other, *fixtures, manifest, sha(manifest), adapter_path,
                    order_salt, output_salt, commitments, keys / "proving.key", refused,
                    reject="private prefix opening does not match",
                )
                wrong = work / "wrong-template"
                wrong.mkdir()
                wrong_order, wrong_output, wrong_packet, _, _, _ = issue_and_seal(
                    "wrong-template", profiles[other][0], manifest, adapter_path, wrong,
                )
                controls["incompatible_order"] = call(
                    "incompatible-order", "pair-prove", other, *fixtures, manifest, sha(manifest), adapter_path,
                    wrong_order, wrong_output, wrong_packet, keys / "proving.key", refused,
                    reject="private prefix circuit rejected the release statement",
                )
                assert not refused.exists()
                invalid_order = [other, manifest, sha(manifest), adapter_path, wrong_order, wrong_output, wrong_packet]

        if compare_key_reuse:
            packed = work / "pair.proofs"
            first_args, second_args = (order[0] for order in batch_orders)
            shared = call("shared-key", "pair-prove-two", keys / "proving.key", packed,
                          *fixtures, *first_args, *second_args)
            raw = packed.read_bytes()
            assert len(raw) == 256 and shared["reported"]["metrics"]["proof_count"] == 2
            shared["verifications"] = []
            for ordinal, (_, commitments, pin, folder) in enumerate(batch_orders):
                proof = folder / "shared-key.proof"
                proof.write_bytes(raw[ordinal * 128: (ordinal + 1) * 128])
                accepted = verify_proof(f"shared-key-{ordinal + 1}-arbiter", commitments, proof, pin)
                assert accepted["reported"]["metrics"]["accepted"]
                shared["verifications"].append(accepted)
            refused = work / "refused-pair.proofs"
            controls["invalid_second_order"] = call(
                "invalid-second-order", "pair-prove-two", work / "deliberately-absent.key", refused,
                *fixtures, *first_args, *invalid_order,
                reject="private prefix circuit rejected the release statement",
            )
            assert not refused.exists()

        first, second = (public / f"order-{i}" for i in (1, 2))
        controls["cross_order_proof"] = call(
            "cross-order", "verify", second / "round.commitments", captures[1]["order_commitment_sha256"],
            vk, vk_pin, first / "round.proof", reject="Groth16 rejected",
        )
        prior = results / "private-prefix"
        legacy = json.loads((prior / "report.json").read_text())
        first_prior = prior / "order-1"
        legacy_result = call(
            "prior-single-template", "verify", first_prior / "round.commitments",
            legacy["orders"][0]["order_commitment_sha256"], prior / "verifying.key",
            legacy["setup"]["reported"]["metrics"]["verifying_key_sha256"],
            first_prior / "round.proof",
        )
        assert legacy_result["reported"]["metrics"]["accepted"]
    expected = {"verifying.key", "order-1/round.commitments", "order-1/round.proof",
                "order-2/round.commitments", "order-2/round.proof"}
    if shared is not None:
        expected.update({"order-1/shared-key.proof", "order-2/shared-key.proof"})
    assert {p.relative_to(public).as_posix() for p in public.rglob("*") if p.is_file()} == expected
    report = {
        "shape": shape_result, "setup": setup, "orders": captures, "refusals": controls,
        "prior_single_template_verification": legacy_result, "arbiter_image": image_id,
        "distinct_templates": len(profiles), "verifying_key_sha256": vk_pin,
        "public_bytes": sum((public / name).stat().st_size for name in expected),
    }
    if shared is not None:
        report["shared_proving_key"] = shared
    (public / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=lambda p: Path(p).resolve())
    parser.add_argument("output", type=lambda p: Path(p).resolve())
    parser.add_argument("--image", default="ubuntu:24.04")
    parser.add_argument("--compare-key-reuse", action="store_true")
    args = parser.parse_args()
    run(args.executable, args.output, args.image, args.compare_key_reuse)
