#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from common_sparse_bf16_proxy import DEBUG_ROOT, FAKE_ROOT, LOCAL_ERROR_METRIC, f, read_csv, selected_to_text, write_csv


SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"
METHOD = "sparse_nvfp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design sparse NVFP4 structural scenarios matched on raw local error.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=30000)
    parser.add_argument("--raw-window-frac", type=float, default=0.006)
    parser.add_argument("--max-neighbors", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rows = load_local_rows(args)
    fit = json.loads((args.output_root / "model" / "fitted_sparse_nvfp4_proxy.json").read_text())
    modules = [
        {
            "name": row["module_name"],
            "layer": int(f(row, "layer")),
            "type": row["module_type"],
            "error": f(row, args.metric),
        }
        for row in local_rows
    ]

    out = args.output_root / "structural_scenarios"
    coefficient_report = analyze_coefficients(modules, fit)
    candidates = sample_candidates(modules, fit, count=args.count, n=args.candidates, seed=args.seed)
    pairs = select_pairs(candidates, pairs=args.pairs, raw_window_frac=args.raw_window_frac, max_neighbors=args.max_neighbors)
    policy_rows, pair_rows = serialize_pairs(pairs, args.count)

    write_csv(out / "sparse_nvfp4_structural_scenario_policies.csv", policy_rows)
    write_csv(out / "sparse_nvfp4_structural_scenario_pairs.csv", pair_rows)
    write_report(out / "sparse_nvfp4_structural_scenario_summary.md", coefficient_report, pair_rows, args)
    plot_pairs(pair_rows, out / "sparse_nvfp4_structural_scenario_pairs.png")
    print(f"wrote {out / 'sparse_nvfp4_structural_scenario_summary.md'}")


def load_local_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.source_015_root / "sensitivity" / "module_method_kernel_local_errors.csv"
    rows = [row for row in read_csv(path) if row.get("method") == METHOD]
    if not rows:
        raise RuntimeError(f"no {METHOD} rows in {path}")
    return rows


def analyze_coefficients(modules: list[dict[str, Any]], fit: dict[str, Any]) -> dict[str, Any]:
    by_layer: dict[int, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    for row in modules:
        by_layer.setdefault(row["layer"], []).append(row["error"])
        by_type.setdefault(row["type"], []).append(row["error"])

    layer_rows = []
    for layer, values in sorted(by_layer.items()):
        layer_rows.append({"name": str(layer), "mean_error": mean(values), "coef": float(fit["layer_coef"][str(layer)])})
    type_rows = []
    for typ, values in sorted(by_type.items()):
        type_rows.append({"name": typ, "mean_error": mean(values), "coef": float(fit["type_coef"][typ])})

    return {
        "layer_rows": layer_rows,
        "type_rows": type_rows,
        "layer_pearson": pearson([row["mean_error"] for row in layer_rows], [row["coef"] for row in layer_rows]),
        "layer_spearman": spearman([row["mean_error"] for row in layer_rows], [row["coef"] for row in layer_rows]),
        "type_pearson": pearson([row["mean_error"] for row in type_rows], [row["coef"] for row in type_rows]),
        "type_spearman": spearman([row["mean_error"] for row in type_rows], [row["coef"] for row in type_rows]),
    }


def sample_candidates(modules: list[dict[str, Any]], fit: dict[str, Any], *, count: int, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: dict[str, dict[str, Any]] = {}
    for _ in range(n):
        selected = rng.sample(modules, count)
        names = sorted(row["name"] for row in selected)
        key = selected_to_text(names)
        if key in out:
            continue
        raw = sum(row["error"] for row in selected)
        structural = sum(row["error"] * coef(fit, row["layer"], row["type"]) for row in selected)
        layer_score = sum(row["error"] * float(fit["layer_coef"][str(row["layer"])]) for row in selected)
        type_score = sum(row["error"] * float(fit["type_coef"][row["type"]]) for row in selected)
        out[key] = {
            "names": names,
            "raw": raw,
            "structural": structural,
            "layer_score": layer_score,
            "type_score": type_score,
            "composition": composition(selected),
        }
    return sorted(out.values(), key=lambda item: item["raw"])


def coef(fit: dict[str, Any], layer: int, typ: str) -> float:
    return float(fit["layer_coef"][str(layer)]) * float(fit["type_coef"][typ])


def composition(selected: list[dict[str, Any]]) -> str:
    layers = {"early": 0, "middle": 0, "late": 0}
    types: dict[str, int] = {}
    for row in selected:
        if row["layer"] <= 7:
            layers["early"] += 1
        elif row["layer"] <= 23:
            layers["middle"] += 1
        else:
            layers["late"] += 1
        types[row["type"]] = types.get(row["type"], 0) + 1
    layer_text = ",".join(f"{key}:{value}" for key, value in layers.items())
    type_text = ",".join(f"{key}:{types[key]}" for key in sorted(types))
    return f"{layer_text};{type_text}"


def select_pairs(
    candidates: list[dict[str, Any]],
    *,
    pairs: int,
    raw_window_frac: float,
    max_neighbors: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    raw_values = [item["raw"] for item in candidates]
    window = max((max(raw_values) - min(raw_values)) * raw_window_frac, 1e-12)
    scored = []
    for i, left in enumerate(candidates):
        j = i + 1
        checked = 0
        while j < len(candidates) and candidates[j]["raw"] - left["raw"] <= window:
            right = candidates[j]
            raw_rel = abs(right["raw"] - left["raw"]) / max((right["raw"] + left["raw"]) / 2.0, 1e-12)
            structural_gap = abs(right["structural"] - left["structural"])
            layer_gap = abs(right["layer_score"] - left["layer_score"])
            type_gap = abs(right["type_score"] - left["type_score"])
            score = structural_gap + 0.35 * layer_gap + 0.35 * type_gap - raw_rel
            low, high = (left, right) if left["structural"] <= right["structural"] else (right, left)
            scored.append((score, low, high, raw_rel))
            j += 1
            checked += 1
            if checked >= max_neighbors:
                break
    selected = []
    used = set()
    for _, low, high, _ in sorted(scored, key=lambda item: item[0], reverse=True):
        low_key = selected_to_text(low["names"])
        high_key = selected_to_text(high["names"])
        if low_key in used or high_key in used:
            continue
        selected.append((low, high))
        used.update([low_key, high_key])
        if len(selected) >= pairs:
            break
    if len(selected) < pairs:
        raise RuntimeError(f"found only {len(selected)} pairs")
    return selected


def serialize_pairs(pairs: list[tuple[dict[str, Any], dict[str, Any]]], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy_rows = []
    pair_rows = []
    for idx, (low, high) in enumerate(pairs):
        pair_id = f"sparse_nvfp4_struct_c{count:03d}_pair{idx:02d}"
        for arm, item in (("low_structural", low), ("high_structural", high)):
            policy_rows.append(
                {
                    "policy_id": f"{pair_id}_{arm}",
                    "pair_id": pair_id,
                    "arm": arm,
                    "sample_kind": "structural_raw_matched",
                    "method": METHOD,
                    "selected_modules": count,
                    "selected_names": selected_to_text(item["names"]),
                    "raw_error_sum": item["raw"],
                    "structural_proxy_sum": item["structural"],
                    "layer_score_sum": item["layer_score"],
                    "type_score_sum": item["type_score"],
                    "composition": item["composition"],
                }
            )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "selected_modules": count,
                "low_raw": low["raw"],
                "high_raw": high["raw"],
                "raw_abs_delta": high["raw"] - low["raw"],
                "raw_rel_gap": abs(high["raw"] - low["raw"]) / max((high["raw"] + low["raw"]) / 2.0, 1e-12),
                "structural_delta": high["structural"] - low["structural"],
                "layer_score_delta": high["layer_score"] - low["layer_score"],
                "type_score_delta": high["type_score"] - low["type_score"],
                "low_composition": low["composition"],
                "high_composition": high["composition"],
            }
        )
    return policy_rows, pair_rows


def write_report(path: Path, coef_report: dict[str, Any], pair_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Sparse NVFP4 Structural Scenario Summary",
        "",
        f"Count: `{args.count}`; pairs: `{len(pair_rows)}`; raw window frac: `{args.raw_window_frac}`.",
        "",
        "## Coefficient vs Local Error",
        "",
        f"- Layer coef vs mean local error: Pearson `{coef_report['layer_pearson']:.4f}`, Spearman `{coef_report['layer_spearman']:.4f}`",
        f"- Type coef vs mean local error: Pearson `{coef_report['type_pearson']:.4f}`, Spearman `{coef_report['type_spearman']:.4f}`",
        "",
        "## Candidate Pair Quality",
        "",
        "| metric | mean | max |",
        "|---|---:|---:|",
    ]
    for key in ("raw_rel_gap", "structural_delta", "layer_score_delta", "type_score_delta"):
        values = [abs(f(row, key)) for row in pair_rows]
        lines.append(f"| {key} | {mean(values):.6f} | {max(values):.6f} |")
    lines.extend(
        [
            "",
            "## Top Pairs",
            "",
            "| pair | raw rel gap | structural delta | layer delta | type delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows[:12]:
        lines.append(
            f"| {row['pair_id']} | {f(row, 'raw_rel_gap'):.6f} | {f(row, 'structural_delta'):.6f} | "
            f"{f(row, 'layer_score_delta'):.6f} | {f(row, 'type_score_delta'):.6f} |"
        )
    lines.extend(["", "## Plot", "", f"- `{path.parent / 'sparse_nvfp4_structural_scenario_pairs.png'}`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_pairs(pair_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([f(row, "raw_rel_gap") for row in pair_rows], [f(row, "structural_delta") for row in pair_rows], alpha=0.85)
    ax.set_xlabel("Raw local relative gap")
    ax.set_ylabel("Structural proxy delta")
    ax.set_title("Sparse NVFP4 raw-matched structural scenario pairs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / den_x / den_y if den_x and den_y else math.nan


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


if __name__ == "__main__":
    main()
