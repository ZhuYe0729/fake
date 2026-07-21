#!/usr/bin/env python3
"""Freeze Llama3.1-tokenized WikiText blocks for 2048+80 runtime NLL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=86)
    args = parser.parse_args()
    sequence_length = 2048 + 80
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                        cache_dir="/home/agent/wja/.cache/huggingface")
    tokens = tokenizer("\n\n".join(data["text"]), return_tensors="pt",
                       add_special_tokens=False).input_ids[0]
    generator = torch.Generator().manual_seed(args.seed)
    starts = torch.randint(0, len(tokens) - sequence_length - 1, (args.blocks,),
                           generator=generator).tolist()
    blocks = torch.stack([tokens[start:start + sequence_length] for start in starts])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blocks, args.output)
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1/train",
        "model": str(MODEL), "blocks": args.blocks, "seed": args.seed,
        "input_tokens": 2048, "decode_tokens": 80, "starts": starts,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
