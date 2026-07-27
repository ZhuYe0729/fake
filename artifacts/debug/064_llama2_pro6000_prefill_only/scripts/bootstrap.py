#!/usr/bin/env python3
"""Freeze policies and deterministically regenerate the missing WikiText tensor."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from common import INPUTS, MODEL, RUN, sha256, write_json

EXPECTED_SAMPLE_SHA256 = "4c859a5b657834d501ba08b1e212c92dc0a7aec638e9ec67437caf11fc0f52dc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-arrow", type=Path, default=Path(
        __import__("os").environ.get("COSPAQ_WIKITEXT_ARROW", "")))
    args = parser.parse_args()
    policy_source = INPUTS / "policies"
    policy_target = RUN / "policies/prefill_only"
    sample_target = RUN / "samples/wikitext_2048_targets.pt"
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
        starts = torch.randint(0, len(tokens) - 2049, (100,), generator=torch.Generator().manual_seed(86)).tolist()
        blocks = torch.stack([tokens[start:start + 2049] for start in starts])
        sample_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blocks, sample_target)
    actual_hash = sha256(sample_target)
    if actual_hash != EXPECTED_SAMPLE_SHA256:
        raise RuntimeError(f"sample hash mismatch: {actual_hash} != {EXPECTED_SAMPLE_SHA256}")
    write_json(RUN / "bootstrap_provenance.json", {
        "policies": len(manifest), "sample_shape": [100, 2049], "sample_sha256": actual_hash,
        "dataset_arrow": str(args.dataset_arrow.resolve()), "dataset_arrow_sha256": sha256(args.dataset_arrow),
        "model": str(MODEL), "model_config_sha256": sha256(MODEL / "config.json")})
    print(json.dumps({"run": str(RUN), "policies": 72, "sample_sha256": actual_hash}, indent=2))


if __name__ == "__main__":
    main()
