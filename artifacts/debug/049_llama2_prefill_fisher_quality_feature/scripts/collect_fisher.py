#!/usr/bin/env python3
"""Collect dense-model module-output Fisher sensitivities on WikiText prompts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/049_llama2_prefill_fisher_quality_feature"
MODEL = "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"
SAMPLES = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat/samples/wikitext_2048_targets.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--blocks", type=int, default=8); parser.add_argument("--tokens", type=int, default=256); args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()
    targets = {}
    for layer in range(32):
        base = model.model.layers[layer]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            targets[f"model.layers.{layer}.self_attn.{name}"] = getattr(base.self_attn, name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            targets[f"model.layers.{layer}.mlp.{name}"] = getattr(base.mlp, name)
    captured: dict[str, torch.Tensor] = {}
    def hook(name):
        def save(_, __, output):
            output.retain_grad(); captured[name] = output
        return save
    handles = [module.register_forward_hook(hook(name)) for name, module in targets.items()]
    sums, counts = defaultdict(float), defaultdict(int)
    blocks = torch.load(SAMPLES, map_location="cpu")[:args.blocks, :args.tokens + 1]
    for ids in blocks:
        captured.clear(); model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids[None].cuda(), use_cache=False).logits.float()
        torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[1:].cuda().reshape(-1)).backward()
        for name, output in captured.items():
            sums[name] += float(output.grad.float().square().mean()); counts[name] += 1
    for handle in handles: handle.remove()
    rows = []
    for name in sorted(targets):
        layer = int(name.split(".")[2]); part = name.rsplit(".", 1)[-1]
        rows.append({"module_name": name, "layer": layer, "part": part, "fisher_mean_grad_sq": sums[name] / counts[name], "blocks": args.blocks, "tokens": args.tokens})
    DEBUG.mkdir(parents=True, exist_ok=True)
    with (DEBUG / "module_fisher.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (DEBUG / "metadata.json").write_text(json.dumps({"model": MODEL, "samples": str(SAMPLES), "blocks": args.blocks, "tokens": args.tokens, "definition": "mean squared gradient of dense next-token NLL with respect to module output"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
