#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any


DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
SOURCE_014_ROOT = FAKE_ROOT / "artifacts/debug/014_llama2_prefill_loss_modeling"
SOURCE_014_SCRIPTS = SOURCE_014_ROOT / "scripts"
SOURCE_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy"
METHOD = "sparse_bf16"
LOCAL_ERROR_METRIC = "output_rel_mse"
LINEAR_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LAYERS = tuple(range(32))
POLICY_COUNTS = (4, 8, 16, 32, 64, 112, 168, 224)
POLICIES_PER_COUNT = 15
SELECTED_SEP = ";"

for path in (FAKE_ROOT.parent, FAKE_ROOT, SOURCE_014_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def load_sparse_local_errors(source_014_root: Path = SOURCE_014_ROOT) -> list[dict[str, Any]]:
    path = source_014_root / "sensitivity" / "module_method_local_errors.csv"
    rows = [row for row in read_csv(path) if row.get("method") == METHOD]
    if not rows:
        raise RuntimeError(f"No {METHOD} local error rows found in {path}")
    return rows


def selected_to_text(names: list[str] | set[str]) -> str:
    return SELECTED_SEP.join(sorted(names))


def selected_from_text(value: str) -> set[str]:
    return {item for item in value.split(SELECTED_SEP) if item}


def policy_paths(output_root: Path = DEBUG_ROOT) -> dict[str, Path]:
    return {
        "policies": output_root / "policies" / "sampled_sparse_bf16_policies.csv",
        "loss": output_root / "loss" / "loss_samples_sparse_bf16.csv",
        "model": output_root / "model" / "fitted_sparse_bf16_proxy.json",
        "predictions": output_root / "model" / "predictions_sparse_bf16.csv",
        "metrics": output_root / "model" / "proxy_metrics_sparse_bf16.csv",
        "plot": output_root / "plots" / "holdout_proxy_vs_loss_delta.png",
        "summary": output_root / "summary" / "README.md",
    }


def sample_uniform(names: list[str], count: int, rng: random.Random) -> list[str]:
    return sorted(rng.sample(names, count))


def sample_balanced(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[str]:
    by_cell: dict[tuple[int, str], list[str]] = {}
    for row in rows:
        key = (int(f(row, "layer")), row["module_type"])
        by_cell.setdefault(key, []).append(row["module_name"])
    for values in by_cell.values():
        rng.shuffle(values)

    cells = [(layer, typ) for layer in LAYERS for typ in LINEAR_TYPES if (layer, typ) in by_cell]
    rng.shuffle(cells)
    selected: list[str] = []
    used: set[str] = set()
    cursor = 0
    while len(selected) < count and cells:
        cell = cells[cursor % len(cells)]
        cursor += 1
        values = by_cell[cell]
        while values and values[-1] in used:
            values.pop()
        if not values:
            continue
        name = values.pop()
        used.add(name)
        selected.append(name)
    if len(selected) < count:
        remaining = [row["module_name"] for row in rows if row["module_name"] not in used]
        selected.extend(rng.sample(remaining, count - len(selected)))
    return sorted(selected)


def build_policy_rows(
    local_rows: list[dict[str, Any]],
    *,
    seed: int,
    policies_per_count: int = POLICIES_PER_COUNT,
    counts: tuple[int, ...] = POLICY_COUNTS,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    names = sorted(row["module_name"] for row in local_rows)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for count in counts:
        if count > len(names):
            raise ValueError(f"requested {count} modules, only {len(names)} available")
        idx = 0
        attempts = 0
        while idx < policies_per_count:
            kind = "balanced" if idx % 2 else "uniform"
            selected = sample_balanced(local_rows, count, rng) if kind == "balanced" else sample_uniform(names, count, rng)
            selected_text = selected_to_text(selected)
            attempts += 1
            if selected_text in seen and attempts < policies_per_count * 30:
                continue
            seen.add(selected_text)
            rows.append(
                {
                    "policy_id": f"c{count:03d}_{kind}_{idx:03d}",
                    "sample_kind": kind,
                    "selected_modules": count,
                    "selected_names": selected_text,
                }
            )
            idx += 1
    return rows
