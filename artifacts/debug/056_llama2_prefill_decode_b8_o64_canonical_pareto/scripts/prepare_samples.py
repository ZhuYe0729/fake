#!/usr/bin/env python3
"""Derive deterministic 2048+64 labels from the existing WikiText blocks."""

from __future__ import annotations

import hashlib
import json

import torch

from scenario import EXP, INPUT_TOKENS, OUTPUT_TOKENS


def main() -> None:
    source = EXP / "samples/wikitext_2048_80.pt"
    output = EXP / "samples/wikitext_2048_64.pt"
    blocks = torch.load(source, map_location="cpu", weights_only=True)
    required = INPUT_TOKENS + OUTPUT_TOKENS
    if blocks.ndim != 2 or blocks.shape[1] < required:
        raise ValueError(f"expected [N,>={required}], got {tuple(blocks.shape)}")
    derived = blocks[:, :required].contiguous()
    torch.save(derived, output)
    metadata = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shape": list(derived.shape), "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
