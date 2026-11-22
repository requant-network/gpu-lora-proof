#!/usr/bin/env python3
"""Prove private prefix orders under one key and a bounded hidden template choice."""

import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from tempfile import TemporaryDirectory

from benchmark import ROOT, sha, timed
from private_prefix_trial import COMMAND, SOURCE_PIN, V, private_fixture


def run(executable, out, image, compare_key_reuse=False, three_templates=False, cuda_verifier=None, wrong_h_library=None):
    os.umask(0o077)
    os.environ["RAYON_NUM_THREADS"] = "2"
    out.mkdir()
    public = out / "public"
    public.mkdir()
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True,
    ).strip()
    results = ROOT / "results"
    profiles = [
        (results / "compact-prefix/cpu/shape.json", results / "text-head-origin/cpu", SOURCE_PIN),
        (results / "compact-prefix/shape.json", results / "text-head-v2/mixed-vocabulary",
         "d200e82458d79b44fc577480ba33b1e7fe79b6e935328bd7812cfdd05763c34a"),
    ]
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

    policy = "triple" if three_templates else "pair"
    captures = []
    controls = {}
    cuda_benchmark = None
    with TemporaryDirectory(prefix="prefix-witness-", dir=out) as scratch:
        work = Path(scratch)
        if three_templates:
            fixture = work / "permuted-reference"
            fixture.mkdir()
            source, adapter = references[0]
            order = copy.deepcopy(source)
            for field in ("x", "target"):
                values = order["heldout"][field]
                values[:2 * V.D] = values[V.D:2 * V.D] + values[:V.D]
            manifest, adapter_path = fixture / "manifest.json", fixture / "adapter.json"
            V.save_json(manifest, order)
            V.save_json(adapter_path, adapter)
            cpu_shape = json.loads(profiles[0][0].read_bytes())
            assert cpu_shape["labels"][0] != cpu_shape["labels"][1]
            cpu_shape["labels"] = V.top1(order["heldout"]["target"])
            third_shape = public / "third-shape.json"
            V.save_json(third_shape, cpu_shape)
            profiles.append((third_shape, fixture, sha(manifest)))
            fixtures.extend([third_shape, manifest, sha(manifest), adapter_path])
            references.append((order, adapter))
            controls["duplicate_template"] = call(
                "duplicate-template", "triple-shape", *fixtures[:8], *fixtures[:4],
                reject="triple requires three distinct prefix templates",
            )
        shape_result = call("shape", f"{policy}-shape", *fixtures)
        keys = work / "keys"
        setup = call("setup", f"{policy}-setup", keys, *fixtures)
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
                "--mount", f"type=bind,src={cuda_verifier or executable},dst=/verify-round,readonly",
                image_id, "/verify-round", COMMAND + "verify", "/round.commitments", order_pin,
                "/verifying.key", vk_pin, "/round.proof",
            ]
            return timed(out, tag, verify)

        batch_orders = []
        shared = None
        choices = list(range(len(profiles)))
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
            batch_orders.append(([choice, *order_args], commitments, order_pin, published))
            capture = {"order_commitment_sha256": order_pin, "issue": issue, "seal": seal}
            if cuda_verifier is None:
                capture["prove"] = call(tag + "-prove", f"{policy}-prove", choice, *fixtures, *order_args, keys / "proving.key", proof)
                capture["arbiter"] = verify_proof(tag + "-arbiter", commitments, proof, order_pin)
            capture["open"] = call(tag + "-open", "open-output", adapter_path, output_salt, commitments, order_pin)
            captures.append(capture)
            if ordinal == 1:
                refused = work / "refused.proof"
                other = 0 if choice == len(profiles) - 1 else len(profiles) - 1
                if three_templates:
                    controls["outside_allowlist"] = call(
                        "outside-allowlist", "triple-prove", 3, *fixtures, *order_args,
                        work / "deliberately-absent.key", refused, reject="triple choice must be 0, 1 or 2",
                    )
                controls["route_switch"] = call(
                    "route-switch", f"{policy}-prove", other, *fixtures, manifest, sha(manifest), adapter_path,
                    order_salt, output_salt, commitments, keys / "proving.key", refused,
                    reject="private prefix opening does not match",
                )
                wrong = work / "wrong-template"
                wrong.mkdir()
                wrong_order, wrong_output, wrong_packet, _, _, _ = issue_and_seal(
                    "wrong-template", profiles[other][0], manifest, adapter_path, wrong,
                )
                controls["incompatible_order"] = call(
                    "incompatible-order", f"{policy}-prove", other, *fixtures, manifest, sha(manifest), adapter_path,
                    wrong_order, wrong_output, wrong_packet, keys / "proving.key", refused,
                    reject="private prefix circuit rejected the release statement",
                )
                assert not refused.exists()
                invalid_order = [other, manifest, sha(manifest), adapter_path, wrong_order, wrong_output, wrong_packet]

        if cuda_verifier is not None:
            packed = work / "cuda-batch.proofs"
            command = COMMAND + "triple-bench-trusted-cuda-h"
            orders = [argument for item in batch_orders for argument in item[0]]
            args = [command, keys / "proving.key", vk_pin, packed, *fixtures, *orders]
            cuda_benchmark = timed(out, "cuda-h-private-batch", [executable, *args], timeout=180)
            raw = packed.read_bytes()
            assert len(raw) == 3 * 128
            for index, (_, commitments, pin, folder) in enumerate(batch_orders):
                proof = folder / "round.proof"
                proof.write_bytes(raw[index * 128:(index + 1) * 128])
                captures[index]["arbiter"] = verify_proof(f"cuda-h-order-{index + 1}-arbiter", commitments, proof, pin)
            refused = work / "refused-cuda.proofs"
            controls["invalid_third_cuda_order"] = timed(out, "invalid-third-cuda-order", [
                executable, command, work / "deliberately-absent.key", vk_pin, refused,
                *fixtures, *orders[:-7], *invalid_order,
            ], reject="private prefix circuit rejected the release statement")
            assert not refused.exists()
            if wrong_h_library is not None:
                args[3] = refused
                controls["wrong_cuda_h_point"] = timed(out, "wrong-private-cuda-h", [
                    "env", f"LD_LIBRARY_PATH={wrong_h_library}", executable, *args,
                ], reject="Groth16 rejected", timeout=180)
                assert not refused.exists()

        if compare_key_reuse:
            packed = work / "batch.proofs"
            count = len(batch_orders)
            batch_command = f"{policy}-prove-{'three' if three_templates else 'two'}"
            orders = [argument for item in batch_orders for argument in item[0]]
            shared = call("shared-key", batch_command, keys / "proving.key", packed, *fixtures, *orders)
            raw = packed.read_bytes()
            assert len(raw) == 128 * count
            assert shared["reported"]["metrics"]["proof_count"] == count
            shared["verifications"] = []
            for ordinal, (_, commitments, pin, folder) in enumerate(batch_orders):
                proof = folder / "shared-key.proof"
                proof.write_bytes(raw[ordinal * 128: (ordinal + 1) * 128])
                accepted = verify_proof(f"shared-key-{ordinal + 1}-arbiter", commitments, proof, pin)
                assert accepted["reported"]["metrics"]["accepted"]
                shared["verifications"].append(accepted)
            refused = work / "refused-batch.proofs"
            valid_prefix = orders[:-len(invalid_order)]
            control = "invalid_third_order" if three_templates else "invalid_second_order"
            controls[control] = call(
                control.replace("_", "-"), batch_command, work / "deliberately-absent.key", refused,
                *fixtures, *valid_prefix, *invalid_order,
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
        if three_templates:
            prior_pair = results / "private-template-pair"
            prior_report = json.loads((prior_pair / "report.json").read_text())
            prior_pair_result = call(
                "prior-pair", "verify", prior_pair / "order-1/round.commitments",
                prior_report["orders"][0]["order_commitment_sha256"], prior_pair / "verifying.key",
                prior_report["verifying_key_sha256"], prior_pair / "order-1/round.proof",
            )
            assert prior_pair_result["reported"]["metrics"]["accepted"]
            same_pair_shape = call("prior-pair-shape", "pair-shape", *fixtures[:8])
            assert same_pair_shape["reported"]["metrics"]["constraints"] == 347690
    expected = {"verifying.key"} | {
        f"order-{i}/{name}" for i in range(1, len(profiles) + 1)
        for name in ("round.commitments", "round.proof")
    }
    if three_templates:
        expected.add("third-shape.json")
    if shared is not None:
        expected.update(f"order-{i}/shared-key.proof" for i in range(1, len(profiles) + 1))
    assert {p.relative_to(public).as_posix() for p in public.rglob("*") if p.is_file()} == expected
    report = {
        "shape": shape_result, "setup": setup, "orders": captures, "refusals": controls,
        "prior_single_template_verification": legacy_result, "arbiter_image": image_id,
        "distinct_templates": len(profiles), "verifying_key_sha256": vk_pin,
        "public_bytes": sum((public / name).stat().st_size for name in expected),
    }
    if three_templates:
        report["prior_pair_verification"] = prior_pair_result
        report["prior_pair_shape"] = same_pair_shape
        report["template_sha256"] = [sha(profile[0]) for profile in profiles]
        report["third_shape_reproducible_public_bytes"] = (public / "third-shape.json").stat().st_size
        report["public_bytes"] -= report["third_shape_reproducible_public_bytes"]
    if shared is not None:
        report["shared_proving_key"] = shared
    if cuda_benchmark is not None:
        report["cuda_h_private_batch"] = cuda_benchmark
    (public / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=lambda p: Path(p).resolve())
    parser.add_argument("output", type=lambda p: Path(p).resolve())
    parser.add_argument("--image", default="ubuntu:24.04")
    parser.add_argument("--compare-key-reuse", action="store_true")
    parser.add_argument("--three-templates", action="store_true")
    parser.add_argument("--cuda-h-cpu-verifier", type=lambda p: Path(p).resolve(),
                        help="benchmark trusted CUDA H on three private fixtures using this CPU-only arbiter")
    parser.add_argument("--cuda-h-wrong-library", type=lambda p: Path(p).resolve(),
                        help="directory with a test library that returns a wrong H point")
    args = parser.parse_args()
    if args.cuda_h_wrong_library and not args.cuda_h_cpu_verifier:
        parser.error("--cuda-h-wrong-library requires the trusted CUDA benchmark")
    run(args.executable, args.output, args.image, args.compare_key_reuse,
        args.three_templates or args.cuda_h_cpu_verifier is not None, args.cuda_h_cpu_verifier, args.cuda_h_wrong_library)
