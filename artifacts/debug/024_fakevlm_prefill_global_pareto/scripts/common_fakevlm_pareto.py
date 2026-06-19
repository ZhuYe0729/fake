#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
SOURCE_020_ROOT = FAKE_ROOT / "artifacts/debug/020_fakevlm_uniform_accuracy"
SOURCE_021_ROOT = FAKE_ROOT / "artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed"

SPEED_BATCHES = (1, 2, 4, 8, 16)
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
POLICY_FORMAT = "fakevlm_prefill_global_pareto_v1"
MODEL_NAME = "FakeVLM"

DEFAULT_MODEL_PATH = "/home/agent/wja/data/models/lingcco/fakeVLM"
DEFAULT_TEST_JSON = "/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json"
DEFAULT_IMAGE_ROOT = "/home/agent/wja/data/datasets/lingcco/FakeClue/test/test"

for path in (FAKE_ROOT, SOURCE_020_ROOT, SOURCE_021_ROOT / "code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = list(rows[0])
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def parse_batches(spec: str) -> tuple[int, ...]:
    if spec == "all":
        return SPEED_BATCHES
    return tuple(int(item) for item in spec.split(",") if item.strip())


def parse_methods(spec: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in spec.split(",") if item.strip())
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={METHODS}")
    if "dense_bf16" not in methods:
        raise ValueError("dense_bf16 must be included")
    return methods


def normalize_module_name(name: str) -> str:
    return re.sub(r"^language_model\.", "model.", name)


def layer_index(module_name: str) -> int:
    match = re.search(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else -1


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


def policy_counts(modules: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("selected_method", row.get("backend", ""))) for row in modules)
    return dict(sorted(counts.items()))


def policy_path(output_root: Path, family: str, name: str) -> Path:
    return output_root / "policies" / family / f"{name}.json"


def pareto_policy_path(output_root: Path, batch_size: int, point_index: int, budget: float) -> Path:
    safe_budget = sanitize(f"{budget:.6g}")
    return output_root / "pareto" / f"batch_{batch_size}" / "policies" / f"point_{point_index:03d}_budget_{safe_budget}.json"


def sanitize(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def write_policy(path: Path, *, family: str, modules: list[dict[str, Any]], summary: dict[str, Any] | None = None, scenario: dict[str, Any] | None = None) -> None:
    payload = {
        "policy_format": POLICY_FORMAT,
        "model": MODEL_NAME,
        "family": family,
        "scenario": scenario or {"mode": "prefill_only"},
        "summary": summary or {},
        "modules": modules,
    }
    write_json(path, payload)
    write_csv(path.with_suffix(".csv"), modules)


def load_policy(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    fmt = payload.get("policy_format")
    if fmt != POLICY_FORMAT:
        raise ValueError(f"unsupported policy format in {path}: {fmt}")
    return payload


def source_021_latency(batch_size: int, source: str = "latency_model") -> list[dict[str, Any]]:
    path = SOURCE_021_ROOT / "candidates" / source / f"batch_{batch_size}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for row in read_csv(path):
        if str(row.get("supported", "")).lower() not in {"true", "1", "yes"}:
            continue
        row = dict(row)
        row["method"] = row.get("backend", "")
        row["batch_size"] = batch_size
        rows.append(row)
    return rows


def source_020_accuracy() -> list[dict[str, Any]]:
    path = SOURCE_020_ROOT / "summary" / "accuracy_summary.csv"
    return read_csv(path) if path.exists() else []


def local_cuda_index(requested_gpu: int) -> int:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required") from exc
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA is required")
    if requested_gpu < count:
        return requested_gpu
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return 0
    raise RuntimeError(f"requested gpu {requested_gpu}, but torch sees {count} CUDA devices")

