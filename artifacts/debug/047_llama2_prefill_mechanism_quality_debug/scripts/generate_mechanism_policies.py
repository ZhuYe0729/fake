#!/usr/bin/env python3
"""Create the fixed 18-policy quant/sparse factorial calibration design."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from common import DEBUG, ERRORS, PARTS, TYPES, sha256


def local_error(method: str) -> dict[str, float]:
    with ERRORS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for layer in range(32):
        for typ, parts in PARTS.items():
            selected = [float(row["local_rel_mse"]) for row in rows if int(row["layer"]) == layer and row["module_type"] in parts and row["method"] == method]
            values[module_name(layer, typ)] = sum(selected) / len(selected)
    return values


def module_name(layer: int, typ: str) -> str:
    parent = "self_attn" if typ in {"qkv_proj", "o_proj"} else "mlp"
    return f"model.layers.{layer}.{parent}.{typ}"


def policy(policy_id: str, family: str, changes: dict[str, str]) -> dict:
    mapping = {module_name(layer, typ): {"prefill_method": changes.get(module_name(layer, typ), "dense_bf16"), "decode_method": "dense_bf16"} for layer in range(32) for typ in TYPES}
    return {"policy_id": policy_id, "scenario": "prefill_only", "policy_kind": family,
            "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16",
            "modules_to_not_convert": ["lm_head"], "method_map": mapping}


def choose(score: dict[str, float], count: int, excluded: set[str] = set()) -> list[str]:
    return [name for name, _ in sorted(((name, value) for name, value in score.items() if name not in excluded), key=lambda item: (item[1], item[0]))[:count]]


def main() -> None:
    output = DEBUG / "policies"
    output.mkdir(parents=True, exist_ok=True)
    q = local_error("dense_nvfp4")
    s = local_error("sparse_bf16")
    sq = local_error("sparse_nvfp4")
    records: list[tuple[str, str, dict[str, str], str]] = []
    for count in (24, 48, 72, 96, 112, 120):
        records.append((f"q{count:03d}", "quant_only", {name: "dense_nvfp4" for name in choose(q, count)}, "train" if count <= 96 else "holdout"))
    for count in (2, 4, 8, 16, 32, 64):
        records.append((f"s{count:03d}", "sparse_only", {name: "sparse_bf16" for name in choose(s, count)}, "train" if count <= 16 else "holdout"))
    for count in (2, 8, 24):
        records.append((f"sq{count:03d}", "compound_colocated", {name: "sparse_nvfp4" for name in choose(sq, count)}, "train" if count < 24 else "holdout"))
    q80 = choose(q, 80)
    for count in (2, 8, 24):
        sparse = choose(s, count, set(q80))
        changes = {name: "dense_nvfp4" for name in q80}
        changes.update({name: "sparse_bf16" for name in sparse})
        records.append((f"qs80_s{count:03d}", "compound_separated", changes, "train" if count < 24 else "holdout"))
    manifest = []
    for policy_id, family, changes, split in records:
        item = policy(policy_id, family, changes)
        path = output / f"{policy_id}.json"
        path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
        counts = {method: sum(value == method for value in changes.values()) for method in ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4")}
        manifest.append({"policy_id": policy_id, "family": family, "split": split, "path": str(path), "sha256": sha256(path), **counts})
    (DEBUG / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"policies": len(manifest), "train": sum(row["split"] == "train" for row in manifest), "holdout": sum(row["split"] == "holdout" for row in manifest)}, indent=2))


if __name__ == "__main__":
    main()
