#!/usr/bin/env python3
"""Choose a fixed feature-space coverage holdout without reading NLL labels."""
from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat"
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def feature(policy: dict) -> torch.Tensor:
    value = torch.zeros((2, len(METHODS), 4, len(TYPES)), dtype=torch.float64)
    for name, methods in policy["method_map"].items():
        bucket = int(name.split(".")[2]) // 8
        raw = name.rsplit(".", 1)[-1]
        typ = "qkv_proj" if raw in {"q_proj", "k_proj", "v_proj"} else "gate_up_proj" if raw in {"gate_proj", "up_proj"} else raw
        for phase_index, phase in enumerate(("prefill", "decode")):
            value[phase_index, METHODS.index(methods[f"{phase}_method"]), bucket, TYPES.index(typ)] += 1
    return value.flatten()


def main() -> None:
    manifest = json.loads((EXP / "policies/prefill_decode/manifest.json").read_text())
    anchors = {"p00", "p01", "p02", "p03", "p04"}
    candidates = [item for item in manifest if item["policy_id"] not in anchors]
    vectors = torch.stack([feature(json.loads(Path(item["path"]).read_text())) for item in candidates])
    vectors = vectors / vectors.norm(dim=1, keepdim=True).clamp(min=1e-12)
    selected: list[int] = [int(torch.argmax(torch.cdist(vectors, vectors.mean(dim=0, keepdim=True))).item())]
    while len(selected) < 18:
        distances = torch.cdist(vectors, vectors[selected]).min(dim=1).values
        distances[selected] = -1
        selected.append(int(torch.argmax(distances).item()))
    holdout = sorted(candidates[index]["policy_id"] for index in selected)
    payload = {"selection": "deterministic farthest-point feature-space coverage; no NLL labels used",
               "anchors_kept_in_train": sorted(anchors), "holdout": holdout,
               "train": [item["policy_id"] for item in manifest if item["policy_id"] not in holdout]}
    output = EXP / "policies/prefill_decode/coverage_holdout.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
