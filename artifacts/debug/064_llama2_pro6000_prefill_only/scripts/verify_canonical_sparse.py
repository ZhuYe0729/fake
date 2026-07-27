#!/usr/bin/env python3
"""Validate that canonical sparse states satisfy their hardware patterns."""
from __future__ import annotations

import json
from pathlib import Path

import torch


from common import RUN, write_json

EXPERIMENT = RUN


def check(state_path: Path, method: str) -> dict[str, int]:
    payload = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
    state = payload["state_dict"]
    checked = 0
    for name, weight in state.items():
        if not name.endswith(".weight") or weight.dim() != 2 or not name.startswith("model.layers."):
            continue
        if method == "sparse_bf16":
            if weight.shape[-1] % 4:
                raise ValueError(f"2:4 shape failure: {name}")
            active = (weight.reshape(-1, 4) != 0).sum(dim=1)
        else:
            if weight.numel() % 8:
                raise ValueError(f"pairwise 4:8 shape failure: {name}")
            active = (weight.reshape(-1, 4, 2).abs().sum(dim=-1) != 0).sum(dim=1)
        if int(active.max()) > 2:
            raise ValueError(f"{method} pattern failure: {name}")
        checked += 1
    return {"checked_linear_weights": checked}


def main() -> None:
    states = {
        "sparse_bf16": EXPERIMENT / "canonical/prepared/sparse_bf16/model.pt",
        "sparse_nvfp4": EXPERIMENT / "canonical/prepared/sparse_nvfp4/model.pt",
    }
    result = {method: check(path, method) for method, path in states.items()}
    result["sparse_nvfp4_final_quantization"] = "deferred_to_phase_exporter"
    output = EXPERIMENT / "canonical/verification.json"
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
