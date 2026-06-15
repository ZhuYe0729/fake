#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


DEBUG_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)
SCENARIO = {
    "name": "normal_02",
    "batch_size": 1,
    "input_tokens": 16384,
    "output_tokens": 256,
    "m_prefill": 16384,
    "m_decode": 1,
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


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
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def layer_weight(row: dict[str, Any]) -> float:
    return {
        "layers_00_07": 1.10,
        "layers_08_15": 1.00,
        "layers_16_23": 1.00,
        "layers_24_31": 1.15,
    }.get(row.get("layer_bucket", ""), 1.0)


def family_weight(row: dict[str, Any]) -> float:
    return {"attention": 1.0, "mlp": 1.0}.get(row.get("module_family", ""), 1.0)


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
    raise ValueError(f"unknown formula: {formula}")


def load_module_quality_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    out: list[dict[str, Any]] = []
    dense_modules = {row["module_name"]: row for row in rows if row.get("method") == "dense_nvfp4"}
    for _module_name, ref in sorted(dense_modules.items(), key=lambda item: int(f(item[1], "module_index"))):
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
            hybrid = dict(row)
            hybrid["method"] = "dense_nvfp4_prefill_marlin_decode"
            hybrid["quality_source_method"] = "dense_nvfp4"
            out.append(hybrid)
    return out


def load_latency_from_pred_candidates(path: Path) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    rows = read_csv(path)
    lookup: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        group = str(row["linear_group"])
        if group == "__TOTAL__":
            continue
        n = int(f(row, "n"))
        k = int(f(row, "k"))
        candidate = str(row["candidate"])
        supported = str(row.get("supported", "")).lower() == "true"
        lookup[(group, n, k, candidate)] = {
            "prefill_ms": f(row, "prefill_ms"),
            "decode_ms": f(row, "decode_ms"),
            "total_ms": f(row, "total_ms"),
            "conversion_ms": f(row, "online_conversion_ms"),
            "prefill_backend": str(row.get("prefill_backend", candidate)),
            "decode_backend": str(row.get("decode_backend", candidate)),
            "supported": supported,
            "reason": str(row.get("reason", "")),
        }
    return lookup


def policy_method_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        method = str(row.get("selected_method", row.get("method", "")))
        counts[method] = counts.get(method, 0) + 1
    return counts


def sanitize(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")
