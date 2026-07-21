#!/usr/bin/env python3
"""Freeze the 100 deterministic WikiText 2048+64 teacher-forcing examples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
DEFAULT_SOURCE = ROOT / "artifacts/debug/044_llama_prefill_decode_vllm_nll/samples/llama31_wikitext_2048_80.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Existing frozen Llama-3.1 2048+80 WikiText blocks.")
    args = parser.parse_args()
    required = 2048 + 64
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    if source.ndim != 2 or source.shape[1] < required:
        raise ValueError(f"expected [N,>={required}] frozen blocks, got {tuple(source.shape)}")
    blocks = source[:, :required].contiguous()
    directory = EXP / "samples"; directory.mkdir(parents=True, exist_ok=True)
    output = directory / "wikitext_2048_64.pt"; torch.save(blocks, output)
    metadata = {"dataset": "Salesforce/wikitext:wikitext-2-raw-v1/train", "model": str(MODEL),
                "source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "shape": list(blocks.shape), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "shape": list(blocks.shape)}))


if __name__ == "__main__":
    main()
