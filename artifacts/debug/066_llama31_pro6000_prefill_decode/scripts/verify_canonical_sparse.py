#!/usr/bin/env python3
"""Validate that canonical sparse states satisfy their hardware patterns."""
from __future__ import annotations

import json
from pathlib import Path

import torch


from common import HISTORICAL_CANONICAL_HASHES, MODEL, RUN, sha256, write_json

EXPERIMENT = RUN


def check(state_path: Path, method: str) -> dict:
    payload = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
    metadata = payload.get("metadata", {})
    if metadata.get("method") != method:
        raise ValueError(f"{method}: metadata method mismatch")
    if metadata.get("selected_modules") != 224 or metadata.get("compressed_modules") != 224:
        raise ValueError(f"{method}: expected 224 compressed modules")
    if metadata.get("skipped"):
        raise ValueError(f"{method}: canonical preparation skipped modules")
    if method == "sparse_nvfp4" and metadata.get("sparse_nvfp4_prequant_only") is not True:
        raise ValueError("sparse_nvfp4 is not prequant-only")
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
    if checked != 224:
        raise ValueError(f"{method}: expected 224 linear weights, got {checked}")
    log = state_path.with_name("compression_log.jsonl")
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    if len(rows) != 224 or any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"{method}: invalid compression log")
    return {"checked_linear_weights": checked, "compression_log_rows": len(rows),
            "selected_modules": metadata["selected_modules"],
            "compressed_modules": metadata["compressed_modules"]}


def main() -> None:
    states = {
        "sparse_bf16": EXPERIMENT / "canonical/prepared/sparse_bf16/model.pt",
        "sparse_nvfp4": EXPERIMENT / "canonical/prepared/sparse_nvfp4/model.pt",
    }
    result = {method: check(path, method) for method, path in states.items()}
    result["sparse_nvfp4_final_quantization"] = "deferred_to_phase_exporter"
    output = EXPERIMENT / "canonical/verification.json"
    write_json(output, result)
    write_json(EXPERIMENT / "canonical/provenance.json", {
        "model": str(MODEL),
        "states": {method: {"path": str(path.resolve()), "sha256": sha256(path),
                            "historical_5090_sha256": HISTORICAL_CANONICAL_HASHES[method]}
                   for method, path in states.items()},
        "historical_hashes_are_informational_only": True,
        "reason": "historical large canonical files are unavailable; regenerated on Pro 6000",
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
