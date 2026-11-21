#!/usr/bin/env python3
"""Train and replay two signed, sharded Qwen LoRA updates."""

import argparse
from fractions import Fraction
from hashlib import sha256 as hash_bytes
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
import torch
from safetensors.torch import load, save
from transformers import AutoModel, AutoTokenizer

from text_head_round import LABELS, V
from verify_text_head_origin import require_digest, sha256


SCHEMA = "qwen-lora-gradient-round/v1"
REVISION = "eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
LAYERS = (22, 23)
WIDTH = 896
SIZE = len(LAYERS) * 2 * WIDTH
SCALE = 1 << 18
RATE_DENOMINATOR = 32
MAX_DELTA = 32
RELATIVE_DENOMINATOR = 256
SLOTS = ("left", "right")
PREFIX = "base_model.model.model.layers."
PROMPT = ("<|im_start|>system\nClassify the following news brief by topic: sports or weather. "
          "Answer with one lowercase word.<|im_end|>\n<|im_start|>user\n{}"
          "<|im_end|>\n<|im_start|>assistant\n")


def initial():
    return [value for layer in LAYERS
            for value in [SCALE // 32 * (1 if (i + layer) % 3 == 0 else -1) for i in range(WIDTH)]
            + [0] * WIDTH]


def hex_string(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


class Order:
    def __init__(self, args):
        self.order, self.pin = V.read_json(args.order, V.MANIFEST_BYTES)
        require_digest(self.pin, args.order_sha256, "round order")
        keys = self.order["keys"]
        if (set(self.order) != {"schema", "round", "keys", "model_files", "corpus_sha256"}
                or self.order["schema"] != SCHEMA or not hex_string(self.order["round"])
                or set(keys) != set(SLOTS) or len(set(keys.values())) != len(SLOTS)
                or not all(hex_string(key) for key in keys.values())
                or set(self.order["model_files"]) != set(MODEL_FILES)
                or not all(hex_string(x) for x in self.order["model_files"].values())):
            raise V.Reject("invalid gradient order")
        self.keys = {slot: Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)) for slot, key in keys.items()}
        require_digest(sha256(args.corpus), self.order["corpus_sha256"], "corpus")
        texts = [line.split("\t", 3) for line in args.corpus.read_text(encoding="ascii").splitlines()]
        adapt = [row for row in texts if row[0] == "adapt"]
        self.heldout = [row for row in texts if row[0] == "heldout"]
        self.shards = dict(zip(SLOTS, (adapt[:8], adapt[8:]), strict=True))
        if ([len(adapt), len(self.heldout)] != [16, 16]
                or len({row[1] for row in texts}) != len(texts)
                or any(len(rows) != 8 or sorted(row[2] for row in rows) != sorted(LABELS * 4)
                       for rows in self.shards.values())
                or sorted(row[2] for row in self.heldout) != sorted(LABELS * 8)):
            raise V.Reject("unexpected shard or validation split")

    def check_model(self, model):
        if {path.name for path in model.iterdir()} != set(MODEL_FILES):
            raise V.Reject("expected exactly the four pinned model files")
        for name, digest in self.order["model_files"].items():
            require_digest(sha256(model / name), digest, name)

    def header(self, slot, previous, prior):
        return {"schema": SCHEMA, "order_sha256": self.pin, "slot": slot,
                "document_ids": [r[1] for r in self.shards[slot]], "step": 2 if prior else 1,
                "input_sha256": V.digest(previous), "prior_phase_sha256": V.digest(prior) if prior else None}

    def merge(self, files, previous, prior=None):
        records = [V.read_json(path, 128 << 10)[0] for path in files]
        if sorted(record["body"]["slot"] for record in records) != list(SLOTS):
            raise V.Reject("expected one signed gradient per slot")
        records.sort(key=lambda record: record["body"]["slot"])
        for record in records:
            body, sig = record["body"], record["signature"]
            slot = body["slot"]
            if set(record) != {"body", "signature"} or not hex_string(sig, 128):
                raise V.Reject("invalid gradient signature encoding")
            self.keys[slot].verify(bytes.fromhex(sig), V.canonical(body))
            header = self.header(slot, previous, prior)
            if type(body["step"]) is not int or body != {**header, "loss": body["loss"], "gradient": body["gradient"]}:
                raise V.Reject("gradient outside the pinned optimizer input")
            g, loss = body["gradient"], body["loss"]
            if not V.bounded_ints(g, SIZE, SCALE * 4096) or type(loss) is not int or not 0 <= loss <= SCALE * 4096:
                raise V.Reject("invalid quantized gradient")
        update = [round(Fraction(left + right, 16 * RATE_DENOMINATOR))
                  for left, right in zip(*(r["body"]["gradient"] for r in records), strict=True)]
        candidate = [p - delta for p, delta in zip(previous, update, strict=True)]
        if not any(update) or any(abs(p) > SCALE * 4 for p in candidate):
            raise V.Reject("empty or out-of-range optimizer step")
        return {"previous": V.digest(previous), "next": V.digest(candidate),
                "contributions": records, "changed": sum(bool(delta) for delta in update)}, candidate


class LowRank(torch.nn.Module):
    def __init__(self, frozen):
        super().__init__()
        self.frozen = frozen
        self.a = torch.nn.Parameter(frozen.weight.new_zeros((1, WIDTH)))
        self.b = torch.nn.Parameter(frozen.weight.new_zeros((WIDTH, 1)))

    def forward(self, x):
        linear = torch.nn.functional.linear
        return self.frozen(x) + linear(linear(x, self.a), self.b)


class Runtime:
    def __init__(self, args, order):
        order.check_model(args.model)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(3)
        torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled = False
        torch.set_float32_matmul_precision("highest")
        self.device = torch.device(args.device)
        if args.device == "cuda" and (torch.cuda.device_count() != 1 or "RTX 5060 Ti" not in torch.cuda.get_device_name()):
            raise V.Reject("expected an isolated RTX 5060 Ti")
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.model = AutoModel.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager",
                                               use_safetensors=True, local_files_only=True).to(self.device).eval().requires_grad_(False)
        if self.model.config.hidden_size != WIDTH or len(self.model.layers) != 24:
            raise V.Reject("expected the pinned transformer shape")
        tags = [tok.encode(label, add_special_tokens=False) for label in LABELS]
        if any(len(ids) != 1 for ids in tags) or tags[0] == tags[1]:
            raise V.Reject("expected distinct single-token labels")
        weights = self.model.get_input_embeddings().weight
        if not self.model.config.tie_word_embeddings:
            raise V.Reject("expected tied token projection")
        self.direction = weights[tags[0][0]] - weights[tags[1][0]]
        self.inputs = {}
        for _, doc, label, text in (*order.shards["left"], *order.shards["right"], *order.heldout):
            tokens = tok.encode(PROMPT.format(text), add_special_tokens=False)
            if not 1 <= len(tokens) <= 256:
                raise V.Reject("text exceeds prompt bounds")
            self.inputs[doc] = (torch.tensor([tokens], device=self.device), 1 if label == LABELS[0] else -1)
        self.params = []
        for i in LAYERS:
            attn = self.model.layers[i].self_attn
            attn.o_proj = LowRank(attn.o_proj)
            self.params.extend([attn.o_proj.a, attn.o_proj.b])

    @torch.no_grad()
    def load_state(self, state):
        for i, parameter in enumerate(self.params):
            values = state[i * WIDTH:(i + 1) * WIDTH]
            parameter.copy_(torch.tensor(values, device=self.device, dtype=torch.float32).reshape(parameter.shape) / SCALE)

    def margin(self, row):
        tokens, target = self.inputs[row[1]]
        hidden = self.model(input_ids=tokens, use_cache=False).last_hidden_state[0, -1]
        margin = torch.dot(hidden, self.direction) * target
        return margin

    def gradient(self, rows):
        total = torch.zeros(SIZE, dtype=torch.float64)
        loss_sum = 0.0
        for row in rows:
            loss = torch.nn.functional.softplus(-self.margin(row))
            grads = torch.autograd.grad(loss, self.params)
            total += torch.cat([g.flatten() for g in grads]).detach().to(device="cpu", dtype=torch.float64)
            loss_sum += loss.item()
        if not torch.isfinite(total).all() or not torch.isfinite(torch.tensor(loss_sum)):
            raise V.Reject("nonfinite objective or gradient")
        return (torch.round(total * SCALE).to(torch.int64).tolist(), round(loss_sum * SCALE))

    @torch.no_grad()
    def heldout(self, rows):
        margins = [self.margin(row).item() for row in rows]
        return {"correct": sum(value > 0 for value in margins),
                "loss": round(torch.nn.functional.softplus(-torch.tensor(margins, dtype=torch.float64)).sum().item() * SCALE)}

    def adapter(self):
        names = [f"{PREFIX}{layer}.self_attn.o_proj.lora_{factor}.weight" for layer in LAYERS for factor in "AB"]
        return save({name: param.detach().cpu() for name, param in zip(names, self.params, strict=True)})


def review_gradient(body, reference, loss):
    deltas = [value - expected for value, expected in zip(body["gradient"], reference, strict=True)]
    maximum = max(map(abs, deltas))
    delta_norm = sum(d * d for d in deltas)
    ref_norm = max(sum(g * g for g in reference), SCALE * SCALE // 10_000)
    loss_delta = abs(body["loss"] - loss)
    if maximum > MAX_DELTA or delta_norm * RELATIVE_DENOMINATOR ** 2 > ref_norm or loss_delta > MAX_DELTA:
        raise V.Reject("signed gradient fails independent numerical replay")
    return {"max_coordinate_ticks": maximum, "sum_squared_delta_ticks": delta_norm,
            "reference_squared_norm_ticks": ref_norm, "loss_delta_ticks": loss_delta}


def run(args):
    started = time.perf_counter()
    order = Order(args)
    genesis = initial()
    first, mid = order.merge(args.first, genesis) if args.first else (None, genesis)
    if args.mode == "worker":
        if args.key is None or args.slot is None or args.second is not None:
            raise V.Reject("worker needs its key and slot")
        key = serialization.load_pem_private_key(args.key.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey) or key.public_key().public_bytes_raw() != order.keys[args.slot].public_bytes_raw():
            raise V.Reject("private key does not match the pinned slot")
        runtime = Runtime(args, order)
        runtime.load_state(mid)
        gradient, loss = runtime.gradient(order.shards[args.slot])
        header = order.header(args.slot, mid, first)
        body = {**header, "gradient": gradient, "loss": loss}
        V.save_json(args.output, {"body": body, "signature": key.sign(V.canonical(body)).hex()})
        return {"slot": args.slot, "step": header["step"], "seconds": time.perf_counter() - started,
                "runtime": torch.__version__, "device": args.device, "record_sha256": sha256(args.output)}

    if args.second is None or first is None or args.key is not None or args.slot is not None or args.device != "cpu":
        raise V.Reject("verification requires both phases on a public CPU profile")
    second, end = order.merge(args.second, mid, first)
    runtime = Runtime(args, order)
    comparisons = []
    for phase, state in ((first, genesis), (second, mid)):
        runtime.load_state(state)
        for record in phase["contributions"]:
            body = record["body"]
            gradient, loss = runtime.gradient(order.shards[body["slot"]])
            comparisons.append({"step": body["step"], "slot": body["slot"], **review_gradient(body, gradient, loss)})
    metrics = []
    for state in (genesis, mid, end):
        runtime.load_state(state)
        metrics.append(runtime.heldout(order.heldout))
    release = metrics[2]["correct"] >= metrics[0]["correct"] and metrics[2]["loss"] < metrics[0]["loss"]
    phases = [first, second]
    adapter = runtime.adapter() if release else None
    if adapter and save(load(adapter)) != adapter:
        raise V.Reject("adapter serialization did not round trip")
    config = V.canonical({
        "base_model_name_or_path": MODEL, "revision": REVISION, "peft_type": "LORA",
        "task_type": "CAUSAL_LM", "inference_mode": True, "r": 1, "lora_alpha": 1,
        "target_modules": ["o_proj"], "layers_to_transform": list(LAYERS), "layers_pattern": "layers",
        "bias": "none", "lora_dropout": 0.0,
    }) + b"\n" if release else None
    certificate = {"schema": SCHEMA, "order_sha256": order.pin, "phases": phases,
                   "merkle_root": V.merkle_root(phases, SCHEMA), "heldout": metrics,
                   "release": release, "adapter_sha256": hash_bytes(adapter).hexdigest() if adapter else None,
                   "adapter_config_sha256": hash_bytes(config).hexdigest() if config else None}
    args.output.mkdir()
    if adapter:
        (args.output / "adapter_model.safetensors").write_bytes(adapter)
        (args.output / "adapter_config.json").write_bytes(config)
    V.save_json(args.output / "certificate.json", certificate)
    return {"seconds": time.perf_counter() - started, "heldout": metrics, "release": release,
            "merkle_root": certificate["merkle_root"], "compared_gradients": comparisons,
            "certificate_sha256": sha256(args.output / "certificate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "verify"))
    for field in ("order", "corpus", "model", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--order-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--slot", choices=SLOTS)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--first", nargs=2, type=Path)
    parser.add_argument("--second", nargs=2, type=Path)
    try:
        summary = run(parser.parse_args())
    except InvalidSignature:
        print("NO: invalid contribution signature", file=sys.stderr)
        return 1
    except (V.Reject, OSError, KeyError, IndexError, TypeError, ValueError, RuntimeError, OverflowError, RecursionError) as error:
        print(f"NO: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
