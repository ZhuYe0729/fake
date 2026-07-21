#!/usr/bin/env python3
"""Generate fixed WikiText blocks and the shared 72-policy prefill design."""
from __future__ import annotations

import argparse
import json
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from common import METHODS, MODELS, TYPES, model_root, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=86)
    return parser.parse_args()


def method_map(index: int) -> dict[str, dict[str, str]]:
    result = {}
    for layer in range(32):
        bucket = layer // 8
        for typ in TYPES:
            if index < 5:
                method = METHODS[index]
                kind = "uniform"
            elif index < 21:
                cell = index - 5
                selected_bucket, selected_type = cell // 4, TYPES[cell % 4]
                method = METHODS[1 + cell % 4] if (bucket, typ) == (selected_bucket, selected_type) else "dense_bf16"
                kind = "controlled_cell"
            elif index < 37:
                bucket_to_change, method_index = divmod(index - 21, 4)
                method = METHODS[1 + method_index] if bucket == bucket_to_change else "dense_bf16"
                kind = "controlled_bucket"
            else:
                severity = (index - 37) % 5
                probabilities = (
                    (0.75, 0.10, 0.03, 0.02, 0.10),
                    (0.55, 0.18, 0.07, 0.04, 0.16),
                    (0.38, 0.25, 0.09, 0.06, 0.22),
                    (0.22, 0.30, 0.13, 0.10, 0.25),
                    (0.08, 0.30, 0.18, 0.16, 0.28),
                )[severity]
                draw, total = random.Random(1307 + index * 100003 + layer * 97 + TYPES.index(typ)).random(), 0.0
                method = METHODS[-1]
                for candidate, probability in zip(METHODS, probabilities):
                    total += probability
                    if draw <= total:
                        method = candidate
                        break
                kind = "balanced_mixed"
            name = f"model.layers.{layer}.self_attn.{typ}" if typ in ("qkv_proj", "o_proj") else f"model.layers.{layer}.mlp.{typ}"
            result[name] = {"prefill_method": method, "decode_method": method}
    return result


def main() -> None:
    args = parse_args()
    root = model_root(args.model)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model]["path"], local_files_only=True, use_fast=True)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir="/home/agent/wja/.cache/huggingface")
    tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(tokens) < args.seq_len + 1:
        raise RuntimeError("WikiText token stream is too short")
    starts = torch.randint(0, len(tokens) - args.seq_len - 1, (args.blocks,), generator=torch.Generator().manual_seed(args.seed)).tolist()
    # Every row contains one unscored leading token and 2048 target tokens.
    blocks = torch.stack([tokens[start : start + args.seq_len + 1] for start in starts])
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    tensor_path = samples / "wikitext_2048_targets.pt"
    torch.save(blocks, tensor_path)
    metadata = {"dataset": "Salesforce/wikitext:wikitext-2-raw-v1/train", "model": str(MODELS[args.model]["path"]), "blocks": args.blocks, "scored_tokens_per_block": args.seq_len, "seed": args.seed, "starts": starts, "tensor_sha256": sha256(tensor_path)}
    (samples / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    policy_dir = root / "policies/prefill_only"
    policy_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index in range(72):
        policy_id = f"p{index:02d}"
        policy = {"policy_id": policy_id, "scenario": "prefill_only", "policy_kind": "uniform" if index < 5 else ("controlled" if index < 37 else "balanced_mixed"), "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16", "modules_to_not_convert": ["lm_head"], "method_map": method_map(index)}
        path = policy_dir / f"{policy_id}.json"
        path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        manifest.append({"policy_id": policy_id, "split": "train" if index < 54 else "holdout", "policy_kind": policy["policy_kind"], "path": str(path), "sha256": sha256(path)})
    (policy_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model": args.model, "sample_shape": list(blocks.shape), "policies": 72, "train": 54, "holdout": 18}, indent=2))


if __name__ == "__main__":
    main()
