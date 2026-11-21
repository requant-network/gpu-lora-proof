#!/usr/bin/env python3
"""Load an accepted adapter through PEFT and check the held-out logits."""

import argparse
import json
from pathlib import Path

import peft
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gradient_round import LABELS, MAX_DELTA, Order, PROMPT, SCALE, V, require_digest, sha256


@torch.no_grad()
def evaluate(model, tokenizer, tags, rows):
    margins = []
    for _, _, label, text in rows:
        tokens = tokenizer.encode(PROMPT.format(text), return_tensors="pt", add_special_tokens=False)
        scores = model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits[0, -1, tags]
        margin = (scores[0] - scores[1]).item()
        margins.append(margin if label == LABELS[0] else -margin)
    return {"correct": sum(x > 0 for x in margins),
            "loss": round(torch.nn.functional.softplus(-torch.tensor(margins, dtype=torch.float64)).sum().item() * SCALE)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("order", "corpus", "model", "release"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-sha256", required=True)
    args = parser.parse_args()
    order = Order(args)
    certificate, _ = V.read_json(args.release / "certificate.json", 128 << 10)
    require_digest(certificate["order_sha256"], order.pin, "certificate order")
    if certificate["release"] is not True:
        raise V.Reject("certificate did not release this adapter")
    weights = args.release / "adapter_model.safetensors"
    config = args.release / "adapter_config.json"
    require_digest(sha256(weights), certificate["adapter_sha256"], "adapter tensors")
    require_digest(sha256(config), certificate["adapter_config_sha256"], "adapter config")
    order.check_model(args.model)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tags = [tokenizer.encode(label, add_special_tokens=False)[0] for label in LABELS]
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager", use_safetensors=True,
                                                local_files_only=True)
    model = PeftModel.from_pretrained(base, args.release, local_files_only=True).eval()
    tensors, loaded = load_file(weights), get_peft_model_state_dict(model, save_embedding_layers=False)
    if set(tensors) != set(loaded) or not all(torch.equal(tensors[k], loaded[k]) for k in tensors):
        raise V.Reject("PEFT loaded different adapter tensors")
    with model.disable_adapter():
        baseline = evaluate(model, tokenizer, tags, order.heldout)
    result = evaluate(model, tokenizer, tags, order.heldout)
    for actual, expected in zip((baseline, result), certificate["heldout"][::2], strict=True):
        if actual["correct"] != expected["correct"] or abs(actual["loss"] - expected["loss"]) > MAX_DELTA:
            raise V.Reject("PEFT held-out evaluation differs from the replay")
    if result["correct"] < baseline["correct"] or result["loss"] >= baseline["loss"]:
        raise V.Reject("PEFT model failed the held-out release gate")
    print(json.dumps({"peft": peft.__version__, "tensors": len(tensors), "baseline": baseline, "adapter": result}))


if __name__ == "__main__":
    main()
