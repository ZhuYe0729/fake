#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from daso.reporting.provenance import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=2112,
                        help="2048 prefill tokens plus up to 64 decode targets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    dataset = Dataset.from_file(str(args.dataset_arrow))
    text = "\n\n".join(row["text"] for row in dataset if row.get("text") and row["text"].strip())
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if tokens.numel() < args.sequence_length:
        raise RuntimeError("dataset is too short")
    rng = random.Random(args.seed)
    starts = [rng.randint(0, tokens.numel() - args.sequence_length) for _ in range(args.samples)]
    blocks = torch.stack([tokens[start:start + args.sequence_length] for start in starts]).long()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blocks, args.output)
    metadata = {"tokenizer": str(args.tokenizer), "dataset_arrow": str(args.dataset_arrow),
                "samples": args.samples, "sequence_length": args.sequence_length,
                "seed": args.seed, "starts": starts, "sha256": sha256(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

