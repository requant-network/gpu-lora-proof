#!/usr/bin/env python3
"""Sign, merge and replay two physical workers' Qwen projection scores."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from text_head_round import V, make_manifest, quantize_features, read_inputs, read_raw_features, source_normalization
from transformer_projection import INPUT_COORDINATES, STEPS, TENSOR, checked_run, evaluate
from verify_text_head_origin import CPU_FILES, require_digest, sha256


SCHEMA = "qwen-projection-shards/v1"
SLOTS = ("left", "right")
ED25519_PUBLIC_PREFIX = "302a300506032b6570032100"


def digest(value):
    return hashlib.sha256(V.canonical(value)).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def candidates(phase, prior):
    if phase == 1:
        return [(step, 0, 0) for step in STEPS]
    step = prior["selection"]["step"]
    if not step:
        raise V.Reject("second phase requires a nonzero output factor")
    return [(step, 0, 0), *((step, coord, sign) for coord in INPUT_COORDINATES for sign in (-1, 1))]


class Round:
    def __init__(self, args):
        self.order, self.pin = V.read_json(args.order, V.MANIFEST_BYTES)
        require_digest(self.pin, args.order_sha256, "round order")
        keys = self.order["keys"]
        if (self.order.get("schema") != SCHEMA or self.order.get("target_tensor") != TENSOR
                or self.order.get("slot_policy") != "contiguous-balanced-eights"
                or not is_sha(self.order["round"]) or set(keys) != set(SLOTS)
                or len(set(keys.values())) != 2
                or not all(re.fullmatch(ED25519_PUBLIC_PREFIX + "[0-9a-f]{64}", key) for key in keys.values())):
            raise V.Reject("unexpected round order")
        self.recipe, pin = V.read_json(args.recipe, V.MANIFEST_BYTES)
        require_digest(pin, self.order["recipe_sha256"], "CPU recipe")
        if (self.recipe.get("schema") != "qwen-head-feature-replay/v1"
                or self.recipe.get("profile") != "cpu" or set(self.recipe["runtime"]) != CPU_FILES):
            raise V.Reject("expected pinned CPU feature recipe")
        self.corpus = args.corpus.resolve()
        require_digest(sha256(self.corpus), self.recipe["corpus_sha256"], "corpus")
        self.texts = [line.split("\t", 3) for line in self.corpus.read_text(encoding="ascii").splitlines()]
        adapt = [row for row in self.texts if row[0] == "adapt"]
        self.shards = dict(zip(SLOTS, (adapt[:8], adapt[8:]), strict=True))
        if any(len(rows) != 8 or sum(r[2] == "sports" for r in rows) != 4 for rows in self.shards.values()):
            raise V.Reject("expected two balanced adaptation shards")
        self.openssl = str(args.openssl.resolve())

    def header(self, phase, slot, prior):
        return {"schema": SCHEMA, "order_sha256": self.pin, "phase": phase, "slot": slot,
                "input_adapter": prior["next"] if prior else self.recipe["model_sha256"],
                "prior_phase_sha256": digest(prior) if prior else None,
                "document_ids": [row[1] for row in self.shards[slot]]}

    def sign(self, body, key):
        with tempfile.TemporaryDirectory(prefix="projection-sign-") as temp:
            message = Path(temp) / "payload"
            message.write_bytes(V.canonical(body))
            sig = subprocess.check_output([self.openssl, "pkeyutl", "-sign", "-rawin",
                                           "-inkey", key.resolve(), "-in", message], timeout=10)
        if len(sig) != 64:
            raise V.Reject("unexpected Ed25519 signature length")
        return {"body": body, "signature": sig.hex()}

    def check_signature(self, envelope):
        body, signature = envelope["body"], envelope["signature"]
        if set(envelope) != {"body", "signature"} or not re.fullmatch("[0-9a-f]{128}", signature):
            raise V.Reject("invalid contribution signature")
        with tempfile.TemporaryDirectory(prefix="projection-sign-") as temp:
            message, public, sig = (Path(temp) / name for name in ("payload", "public", "signature"))
            message.write_bytes(V.canonical(body))
            public.write_bytes(bytes.fromhex(self.order["keys"][body["slot"]]))
            sig.write_bytes(bytes.fromhex(signature))
            reply = subprocess.run([self.openssl, "pkeyutl", "-verify", "-rawin", "-pubin",
                                    "-keyform", "DER", "-inkey", public, "-in", message, "-sigfile", sig],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        if reply.returncode:
            raise V.Reject("contribution signature did not verify")

    def merge(self, phase, files, prior=None):
        records = [V.read_json(path, V.MANIFEST_BYTES)[0] for path in files]
        if sorted(record["body"]["slot"] for record in records) != list(SLOTS):
            raise V.Reject("expected exactly one contribution from each slot")
        records.sort(key=lambda record: record["body"]["slot"])
        grid = candidates(phase, prior)
        totals = [{"step": step, "coordinate": coord, "sign": sign, "correct": 0, "squared_error": 0}
                  for step, coord, sign in grid]
        for envelope in records:
            body = envelope["body"]
            self.check_signature(envelope)
            header = self.header(phase, body["slot"], prior)
            if body != {**header, "candidates": body["candidates"]} or len(body["candidates"]) != len(grid):
                raise V.Reject("contribution does not bind this phase input")
            for item, total in zip(body["candidates"], totals, strict=True):
                if (set(item) != {*total, "adapter_sha256", "features_sha256"}
                        or any(type(item[key]) is not int or item[key] != total[key]
                               for key in ("step", "coordinate", "sign"))
                        or not is_sha(item["features_sha256"]) or not is_sha(item["adapter_sha256"])
                        or type(item["correct"]) is not int or not 0 <= item["correct"] <= 8
                        or type(item["squared_error"]) is not int or not 0 <= item["squared_error"] <= 1 << 44):
                    raise V.Reject("invalid shard candidate result")
                claimed = total.setdefault("adapter_sha256", item["adapter_sha256"])
                require_digest(item["adapter_sha256"], claimed, "paired candidate adapter")
                total["correct"] += item["correct"]
                total["squared_error"] += item["squared_error"]
        if phase == 1:
            selected = min(totals, key=lambda r: (r["squared_error"], abs(r["step"]), r["step"]))
        else:
            selected = min(totals, key=lambda r: (r["squared_error"], abs(r["sign"]), r["coordinate"], r["sign"]))
        return {"previous": prior["next"] if prior else self.recipe["model_sha256"],
                "next": selected["adapter_sha256"], "selection": selected,
                "totals": totals, "contributions": records}


class Replay:
    def __init__(self, args, order, root):
        self.root, self.order = root, order
        recipe = order.recipe
        require_digest(sha256(args.model), recipe["model_sha256"], "GGUF")
        library = args.library.resolve()
        for name, expected in recipe["runtime"].items():
            require_digest(sha256(library / name), expected, name)
        self.env = {"PATH": os.defpath, "LC_ALL": "C", "LD_LIBRARY_PATH": str(library),
                    "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "3"}
        self.probe = [str(args.probe.resolve()), str(args.model.resolve()), str(library)]
        self.writer = str(args.writer.resolve())
        baseline = root / "baseline.tsv"
        self.capture(order.corpus, baseline, None)
        pin = sha256(baseline)
        require_digest(pin, recipe["features_sha256"], "baseline features")
        self.head = make_manifest(read_inputs(order.corpus, baseline), recipe["corpus_sha256"], pin)
        self.centers, self.scales = source_normalization(read_raw_features(order.texts, baseline)["source"])
        self.coefficients = [str(self.head["W"][i * V.D]) for i in range(V.D)]
        if not any(int(coefficient) for coefficient in self.coefficients):
            raise V.Reject("empty output direction")

    def capture(self, corpus, output, adapter):
        command = [*self.probe, str(corpus), "--cpu"]
        if adapter:
            command.append(str(adapter))
        with output.open("wb") as file:
            checked_run(command, self.env, output=file)

    def adapter(self, step, coord, sign):
        if not step:
            return None
        path = self.root / f"adapter-{step}-{coord}-{sign}.gguf"
        if not path.exists():
            addition = [str(coord), str(sign)] if sign else []
            checked_run([self.writer, str(path), str(step), *self.coefficients, *addition], self.env)
        return path

    def score(self, texts, adapter):
        corpus, features = (self.root / name for name in ("split.tsv", "features.tsv"))
        corpus.write_bytes("".join("\t".join(row) + "\n" for row in texts).encode("ascii"))
        self.capture(corpus, features, adapter)
        captured = read_raw_features(texts, features)
        integer = quantize_features(captured, self.centers, self.scales)
        return {**evaluate(self.head, integer[texts[0][0]]), "features_sha256": sha256(features)}

    def body(self, phase, slot, prior):
        header = self.order.header(phase, slot, prior)
        rows = []
        for step, coord, sign in candidates(phase, prior):
            adapter = self.adapter(step, coord, sign)
            rows.append({"step": step, "coordinate": coord, "sign": sign,
                         "adapter_sha256": sha256(adapter) if adapter else self.order.recipe["model_sha256"],
                         **self.score(self.order.shards[slot], adapter)})
        return {**header, "candidates": rows}


def worker(args, order, replay, prior):
    body = replay.body(2 if prior else 1, args.slot, prior)
    signed = order.sign(body, args.key)
    order.check_signature(signed)
    with args.output.open("xb") as out:
        out.write(V.canonical(signed) + b"\n")
    return {"slot": args.slot, "phase": body["phase"], "record_sha256": sha256(args.output)}


def verify(args, order, replay, first, second):
    for phase, record in enumerate((first, second), 1):
        for item in record["contributions"]:
            body = item["body"]
            replayed = replay.body(phase, body["slot"], first if phase == 2 else None)
            if replayed != body:
                raise V.Reject(f"shard {body['slot']} phase {phase} differs from local replay")
    heldout = [row for row in order.texts if row[0] == "heldout"]
    b = replay.adapter(first["selection"]["step"], 0, 0)
    end = second["selection"]
    ab = replay.adapter(end["step"], end["coordinate"], end["sign"])
    baseline = replay.score(heldout, None)
    one, two = replay.score(heldout, b), replay.score(heldout, ab)
    winner, release = None, None
    if one["correct"] >= baseline["correct"] and one["squared_error"] < baseline["squared_error"]:
        winner, release = b, "phase1"
    if (end["sign"] and two["correct"] >= max(one["correct"], baseline["correct"])
            and two["squared_error"] < min(one["squared_error"], baseline["squared_error"])):
        winner, release = ab, "phase2"
    phases = [first, second]
    root = V.merkle_root(phases, SCHEMA)
    certificate = {"schema": SCHEMA, "order_sha256": order.pin, "phases": phases, "merkle_root": root,
                   "heldout": {"baseline": baseline, "phase1": one, "phase2": two},
                   "release": release, "adapter_sha256": sha256(winner) if winner else None}
    args.output.mkdir()
    V.save_json(args.output / "certificate.json", certificate)
    if winner:
        with (args.output / "adapter.gguf").open("xb") as out:
            out.write(winner.read_bytes())
    return {"certificate_sha256": sha256(args.output / "certificate.json"), "merkle_root": root,
            "release": release, "heldout": certificate["heldout"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "verify"))
    for field in ("order", "recipe", "corpus", "model", "library", "probe", "writer", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--order-sha256", required=True)
    parser.add_argument("--openssl", type=Path, default=Path("/usr/bin/openssl"))
    parser.add_argument("--slot", choices=SLOTS)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--first", nargs="+", type=Path)
    parser.add_argument("--second", nargs="+", type=Path)
    started = time.perf_counter()
    try:
        args = parser.parse_args()
        order = Round(args)
        if args.mode == "worker":
            if args.slot is None or args.key is None or args.second is not None:
                raise V.Reject("worker requires a slot and signing key")
            prior = order.merge(1, args.first) if args.first is not None else None
        else:
            if args.first is None or args.second is None or args.slot is not None or args.key is not None:
                raise V.Reject("verifier requires both phase contribution sets")
            prior = order.merge(1, args.first)
            second = order.merge(2, args.second, prior)
        with tempfile.TemporaryDirectory(prefix="requant-projection-shards-") as directory:
            replay = Replay(args, order, Path(directory))
            result = worker(args, order, replay, prior) if args.mode == "worker" else verify(args, order, replay, prior, second)
        print(json.dumps({**result, "seconds": time.perf_counter() - started}))
    except (V.Reject, KeyError, TypeError, ValueError, OSError, OverflowError, RecursionError,
            subprocess.SubprocessError) as error:
        print(f"NO: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
