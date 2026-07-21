#!/usr/bin/env python3
"""Generate additional training-only policies spanning sensitivity bands."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/048_llama2_prefill_quality_coverage"
ERRORS = ROOT / "artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv"
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
PARTS = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "o_proj": ("o_proj",), "gate_up_proj": ("gate_proj", "up_proj"), "down_proj": ("down_proj",)}


def module(layer: int, typ: str) -> str:
    return f"model.layers.{layer}.{'self_attn' if typ in {'qkv_proj', 'o_proj'} else 'mlp'}.{typ}"


def errors(method: str) -> dict[str, float]:
    with ERRORS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {module(layer, typ): sum(float(row["local_rel_mse"]) for row in rows if int(row["layer"]) == layer and row["module_type"] in parts and row["method"] == method) / len(parts) for layer in range(32) for typ, parts in PARTS.items()}


def band(score: dict[str, float], count: int, name: str, excluded: set[str] | None = None) -> list[str]:
    ranked = [item[0] for item in sorted(score.items(), key=lambda item: (item[1], item[0])) if item[0] not in (excluded or set())]
    start = {"mid": (len(ranked) - count) // 2, "high": len(ranked) - count}[name]
    return ranked[start:start + count]


def policy(policy_id: str, family: str, changes: dict[str, str]) -> dict:
    mapping = {module(layer, typ): {"prefill_method": changes.get(module(layer, typ), "dense_bf16"), "decode_method": "dense_bf16"} for layer in range(32) for typ in TYPES}
    return {"policy_id": policy_id, "policy_kind": family, "scenario": "prefill_only", "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16", "modules_to_not_convert": ["lm_head"], "method_map": mapping}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    q, s, sq = errors("dense_nvfp4"), errors("sparse_bf16"), errors("sparse_nvfp4")
    records: list[tuple[str, str, dict[str, str]]] = []
    for count in (24, 48, 80, 96):
        for sensitivity in ("mid", "high"):
            records.append((f"q{count:03d}_{sensitivity}", "quant_sensitivity_coverage", {name: "dense_nvfp4" for name in band(q, count, sensitivity)}))
    for count in (8, 16, 24, 48):
        for sensitivity in ("mid", "high"):
            records.append((f"s{count:03d}_{sensitivity}", "sparse_sensitivity_coverage", {name: "sparse_bf16" for name in band(s, count, sensitivity)}))
    for count in (16, 32):
        for sensitivity in ("mid", "high"):
            records.append((f"sq{count:03d}_{sensitivity}", "colocated_sensitivity_coverage", {name: "sparse_nvfp4" for name in band(sq, count, sensitivity)}))
    for qcount, qband, scount, sband in ((48, "mid", 8, "mid"), (48, "high", 8, "high"), (80, "mid", 16, "mid"), (80, "high", 16, "high")):
        qnames = band(q, qcount, qband); snames = band(s, scount, sband, set(qnames))
        records.append((f"qs{qcount:03d}_{qband}_s{scount:03d}_{sband}", "separated_sensitivity_coverage", {**{name: "dense_nvfp4" for name in qnames}, **{name: "sparse_bf16" for name in snames}}))
    if len(records) != 24:
        raise RuntimeError(len(records))
    output = DEBUG / "policies"; output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for policy_id, family, changes in records:
        path = output / f"{policy_id}.json"; path.write_text(json.dumps(policy(policy_id, family, changes), indent=2, sort_keys=True) + "\n")
        manifest.append({"policy_id": policy_id, "family": family, "split": "train", "path": str(path), "sha256": digest(path), **{method: sum(value == method for value in changes.values()) for method in ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4")}})
    (DEBUG / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"policies": len(manifest), "output": str(DEBUG / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
