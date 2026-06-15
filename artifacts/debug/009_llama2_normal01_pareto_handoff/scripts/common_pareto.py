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

METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)
QUALITY_METHOD_MAP = {
    "dense_bf16": "dense_bf16",
    "dense_nvfp4": "dense_nvfp4",
    "sparse_bf16": "sparse_bf16",
    "sparse_nvfp4": "sparse_nvfp4",
    "marlin_nvfp4": "dense_nvfp4",
    "dense_nvfp4_prefill_marlin_decode": "dense_nvfp4",
}
DECODE_METHOD_MAP = {
    "dense_bf16": "dense_bf16",
    "dense_nvfp4": "dense_nvfp4",
    "sparse_bf16": "dense_bf16",
    "sparse_nvfp4": "dense_bf16",
    "marlin_nvfp4": "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode": "marlin_nvfp4",
}
SCENARIO = {
    "name": "normal_01",
    "batch_size": 1,
    "input_tokens": 16384,
    "output_tokens": 32,
    "m_prefill": 16384,
    "m_decode": 1,
}
POLICY_JSON_PATH = (
    FAKE_ROOT / "artifacts/results/benchmarks/hybrid/pred/normal_01/"
    "llama_2_7b_normal_01_policy.json"
)

NVFP4_CONVERSIONS = {
    "dense_nvfp4": "canonical_to_cutlass",
    "marlin_nvfp4": "canonical_to_marlin",
}
COMPATIBLE_PAIRS = {
    ("dense_nvfp4", "marlin_nvfp4"),
    ("marlin_nvfp4", "dense_nvfp4"),
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
            hybrid = dict(row)
            hybrid["method"] = "dense_nvfp4_prefill_marlin_decode"
            hybrid["quality_source_method"] = "dense_nvfp4"
            out.append(hybrid)
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


def is_legal_strategy(prefill_backend: str, decode_backend: str) -> bool:
    return prefill_backend == decode_backend or (prefill_backend, decode_backend) in COMPATIBLE_PAIRS


def load_latency_from_policy_json(policy_path: Path = POLICY_JSON_PATH) -> dict[str, dict[str, Any]]:
    """Parse policy JSON into a per-module-group latency lookup.

    Returns a dict keyed by (linear_group, n, k) with values containing
    prefill_latency_by_kernel, decode_latency_by_kernel, conversion_latency_by_type.
    """
    data = read_json(policy_path)
    lookup: dict[str, dict[str, Any]] = {}
    for module in data.get("modules", []):
        name = str(module["name"])
        n = int(module["n"])
        k = int(module["k"])
        count = int(module.get("count", 1))
        key = (name, n, k)
        prefill = {}
        for c in module.get("prefill_candidates", []):
            if c.get("supported"):
                prefill[str(c["kernel"])] = float(c["latency_ms"])
        decode = {}
        for c in module.get("decode_candidates", []):
            if c.get("supported"):
                decode[str(c["kernel"])] = float(c["latency_ms"])
        conversion = {}
        for c in module.get("conversion_candidates", []):
            if c.get("supported"):
                conversion[str(c["conversion"])] = float(c["latency_ms"])
        lookup[key] = {
            "prefill": prefill,
            "decode": decode,
            "conversion": conversion,
            "count": count,
            "name": name,
            "n": n,
            "k": k,
        }
    return lookup


def get_latency_for_module(
    module_name: str,
    method: str,
    latency_lookup: dict[str, dict[str, Any]],
    output_tokens: int = 32,
) -> dict[str, Any]:
    """Get prefill_ms, decode_ms, conversion_ms, total_ms for a module+method."""
    group = normalize_group_name(module_name)
    quality_row_sample = None
    rows = load_module_quality_rows()
    for row in rows:
        if row.get("module_name") == module_name and row.get("method") == "dense_bf16":
            quality_row_sample = row
            break
    if quality_row_sample is None:
        n_val, k_val = 4096, 4096
    else:
        n_val = int(f(quality_row_sample, "out_features"))
        k_val = int(f(quality_row_sample, "in_features"))

    key = (group, n_val, k_val)
    if key not in latency_lookup:
        alt_key = None
        for lk in latency_lookup:
            if lk[0] == group:
                alt_key = lk
                break
        if alt_key is None:
            raise KeyError(f"no latency data for {module_name} (group={group}, n={n_val}, k={k_val})")
        key = alt_key

    lat = latency_lookup[key]
    prefill_backend = method
    if method == "dense_nvfp4_prefill_marlin_decode":
        prefill_backend = "dense_nvfp4"
    decode_backend = DECODE_METHOD_MAP[method]

    prefill_ms = lat["prefill"].get(prefill_backend, 0.0)
    decode_ms_val = lat["decode"].get(decode_backend, 0.0)

    conversion_ms = 0.0
    if method == "dense_nvfp4":
        conversion_ms = lat["conversion"].get("canonical_to_cutlass", 0.0)
    elif method in ("marlin_nvfp4", "dense_nvfp4_prefill_marlin_decode"):
        conversion_ms = lat["conversion"].get("canonical_to_marlin", 0.0)

    total_ms = prefill_ms + output_tokens * decode_ms_val + conversion_ms

    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms_val,
        "conversion_ms": conversion_ms,
        "total_ms": total_ms,
        "prefill_backend": prefill_backend,
        "decode_backend": decode_backend,
    }
