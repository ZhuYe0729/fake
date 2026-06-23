#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
SOURCE_024_ROOT = FAKE_ROOT / "artifacts/debug/024_fakevlm_prefill_global_pareto"
SOURCE_020_ROOT = FAKE_ROOT / "artifacts/debug/020_fakevlm_uniform_accuracy"
SOURCE_021_CODE = FAKE_ROOT / "artifacts/debug/021_fakevlm_linear_hybrid_prefill_speed/code"

METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
DEFAULT_BATCH_SIZE = 16
DEFAULT_SUBSET_SEED = 20260619
DEFAULT_SUBSET_FRACTION = 0.2
DEFAULT_MODEL_PATH = "/home/agent/wja/data/models/lingcco/fakeVLM"
DEFAULT_TEST_JSON = "/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json"
DEFAULT_IMAGE_ROOT = "/home/agent/wja/data/datasets/lingcco/FakeClue/test/test"

for path in (FAKE_ROOT, SOURCE_024_ROOT / "scripts", SOURCE_020_ROOT, SOURCE_021_CODE):
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


def cost_rows(batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, Any]]:
    path = SOURCE_024_ROOT / "costs" / f"batch_{batch_size}" / "module_method_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv(path)


def selected_024_rows(batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, Any]]:
    path = SOURCE_024_ROOT / "validation" / "selected_pareto_points.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return [row for row in read_csv(path) if int(f(row, "batch_size")) == batch_size]


def report_024_rows(batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict[str, Any]]:
    path = SOURCE_024_ROOT / "report" / "final_fakevlm_report.csv"
    if path.exists():
        return [row for row in read_csv(path) if row.get("row_type") == "pareto" and int(f(row, "batch_size")) == batch_size]
    return []
