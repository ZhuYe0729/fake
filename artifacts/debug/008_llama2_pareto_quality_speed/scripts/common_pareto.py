#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = FAKE_ROOT.parent
QUALITY_ROOT = DEBUG_ROOT.parent / "007_llama2_quality_modeling"
SOURCE_003_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy"
ORACLE_SUMMARY_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_oracle_summary"

for path in (WORKSPACE_ROOT, FAKE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")
QUALITY_METHOD_MAP = {
    "dense_bf16": "dense_bf16",
    "dense_nvfp4": "dense_nvfp4",
    "sparse_bf16": "sparse_bf16",
    "sparse_nvfp4": "sparse_nvfp4",
    "marlin_nvfp4": "dense_nvfp4",
}
SCENARIO = {
    "name": "prefill_only",
    "batch_size": 16,
    "input_tokens": 1024,
    "output_tokens": 0,
    "m_prefill": 16384,
}


@dataclass(frozen=True)
class LinearGroup:
    name: str
    n: int
    k: int
    count: int


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def normalize_group_name(module_name: str) -> str:
    name = re.sub(r"^(model\.)?layers\.\d+\.", "", module_name)
    name = re.sub(r"^(model\.)?language_model\.layers\.\d+\.", "", name)
    return name


def layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2])
    return -1


def module_type(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def module_family(module_name: str) -> str:
    typ = module_type(module_name)
    if typ in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attention"
    if typ in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    return "other"


def layer_bucket(layer: int) -> str:
    if 0 <= layer <= 7:
        return "layers_00_07"
    if 8 <= layer <= 15:
        return "layers_08_15"
    if 16 <= layer <= 23:
        return "layers_16_23"
    if 24 <= layer <= 31:
        return "layers_24_31"
    return "other"


def layer_weight(row: dict[str, Any]) -> float:
    return {
        "layers_00_07": 1.10,
        "layers_08_15": 1.00,
        "layers_16_23": 1.00,
        "layers_24_31": 1.15,
    }.get(row.get("layer_bucket", ""), 1.0)


def family_weight(row: dict[str, Any]) -> float:
    return {
        "attention": 1.0,
        "mlp": 1.0,
    }.get(row.get("module_family", ""), 1.0)


def quality_cost(row: dict[str, Any], formula: str) -> float:
    if row.get("method") == "dense_bf16":
        return 0.0
    local = f(row, "local_rel_mse")
    numel = max(f(row, "numel"), 0.0)
    if formula == "local_rel_mse":
        return local
    if formula == "local_rel_mse_log_numel":
        return local * math.log1p(numel)
    if formula == "local_rel_mse_log_numel_layer_family":
        return local * math.log1p(numel) * layer_weight(row) * family_weight(row)
    if formula == "local_rel_mse_log_numel_activation_outlier":
        return local * math.log1p(numel) * (1.0 + f(row, "activation_outlier_ratio_6x_weight_mean"))
    if formula == "local_rel_mse_log_numel_weight_outlier":
        return local * math.log1p(numel) * (1.0 + f(row, "weight_outlier_ratio_6x"))
    raise ValueError(f"unknown formula: {formula}")


def load_module_quality_rows(quality_root: Path = QUALITY_ROOT) -> list[dict[str, Any]]:
    path = quality_root / "sensitivity" / "module_method_errors.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    out: list[dict[str, Any]] = []
    dense_modules = {row["module_name"]: row for row in rows if row.get("method") == "dense_nvfp4"}
    for module_name, ref in sorted(dense_modules.items(), key=lambda item: int(item[1]["module_index"])):
        dense = dict(ref)
        dense["method"] = "dense_bf16"
        dense["local_rel_mse"] = 0.0
        dense["local_rmse_over_rms"] = 0.0
        dense["local_max_abs_error"] = 0.0
        out.append(dense)
    out.extend(rows)
    for row in rows:
        if row.get("method") == "dense_nvfp4":
            marlin = dict(row)
            marlin["method"] = "marlin_nvfp4"
            marlin["quality_source_method"] = "dense_nvfp4"
            out.append(marlin)
    return out


def linear_groups_from_quality(rows: Iterable[dict[str, Any]]) -> list[LinearGroup]:
    groups: dict[tuple[str, int, int], int] = {}
    seen_modules: set[str] = set()
    for row in rows:
        if row.get("method") != "dense_bf16":
            continue
        name = str(row["module_name"])
        if name in seen_modules:
            continue
        seen_modules.add(name)
        group = normalize_group_name(name)
        n = int(f(row, "out_features"))
        k = int(f(row, "in_features"))
        groups[(group, n, k)] = groups.get((group, n, k), 0) + 1
    return [LinearGroup(name, n, k, count) for (name, n, k), count in sorted(groups.items())]


def policy_method_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        method = str(row.get("selected_method", row.get("method", "")))
        counts[method] = counts.get(method, 0) + 1
    return counts


def sanitize(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")
