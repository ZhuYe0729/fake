#!/usr/bin/env python3
"""Freeze local policies and deterministically build 2048+64 WikiText samples."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from common import INPUTS, MODEL, RUN, sha256, write_json

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-arrow", type=Path, default=Path(
        __import__("os").environ.get("COSPAQ_WIKITEXT_ARROW", "")))
    args = parser.parse_args()
    policy_source = INPUTS / "policies"
    policy_target = RUN / "policies/prefill_decode"
    sample_target = RUN / "samples/wikitext_2048_64.pt"
    if RUN.exists() and not args.resume:
        raise FileExistsError(f"run exists; use --resume: {RUN}")
    policy_target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in sorted(policy_source.glob("p[0-9][0-9].json")):
        target = policy_target / source.name
        if target.exists() and sha256(target) != sha256(source):
            raise RuntimeError(f"policy differs: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        policy = json.loads(target.read_text())
        index = int(source.stem[1:])
        manifest.append({"policy_id": source.stem, "split": "train" if index < 54 else "holdout",
                         "policy_kind": policy.get("policy_kind", "uniform" if index < 5 else "calibration"),
                         "path": str(target.resolve()), "sha256": sha256(target)})
    if len(manifest) != 72:
        raise RuntimeError(f"expected 72 policies, found {len(manifest)}")
    write_json(policy_target / "manifest.json", manifest)
    if not sample_target.exists():
        if not args.dataset_arrow.is_file():
            raise FileNotFoundError(f"missing WikiText Arrow: {args.dataset_arrow}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
        dataset = Dataset.from_file(str(args.dataset_arrow))
        tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
        required = 2048 + 64
        starts = torch.randint(0, len(tokens) - required, (300,),
                               generator=torch.Generator().manual_seed(86)).tolist()
        blocks = torch.stack([tokens[start:start + required] for start in starts]).contiguous()
        sample_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blocks, sample_target)
    blocks = torch.load(sample_target, map_location="cpu", weights_only=True)
    if tuple(blocks.shape) != (300, 2112):
        raise RuntimeError(f"sample shape mismatch: {tuple(blocks.shape)}")
    actual_hash = sha256(sample_target)
    write_json(RUN / "bootstrap_provenance.json", {
        "policies": len(manifest), "sample_shape": [300, 2112], "sample_sha256": actual_hash,
        "sample_seed": 86, "calibration_blocks": 100,
        "dataset_arrow": str(args.dataset_arrow.resolve()), "dataset_arrow_sha256": sha256(args.dataset_arrow),
        "model": str(MODEL), "model_config_sha256": sha256(MODEL / "config.json")})
    print(json.dumps({"run": str(RUN), "policies": 72, "sample_sha256": actual_hash}, indent=2))


if __name__ == "__main__":
    main()
