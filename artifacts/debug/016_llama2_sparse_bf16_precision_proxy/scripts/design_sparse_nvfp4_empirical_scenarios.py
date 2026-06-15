#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from common_sparse_bf16_proxy import DEBUG_ROOT, FAKE_ROOT, LOCAL_ERROR_METRIC, f, read_csv, selected_from_text, selected_to_text, write_csv


SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"
METHOD = "sparse_nvfp4"
TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BUCKETS = ((0, 7, "L0_7"), (8, 15, "L8_15"), (16, 23, "L16_23"), (24, 31, "L24_31"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design sparse NVFP4 empirical structural scenarios.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-015-root", type=Path, default=SOURCE_015_ROOT)
    parser.add_argument("--metric", default=LOCAL_ERROR_METRIC)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=30000)
    parser.add_argument("--raw-window-frac", type=float, default=0.006)
    parser.add_argument("--max-neighbors", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=["max_gap", "balanced"], default="max_gap")
    parser.add_argument("--target-score-delta", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = load_modules(args)
    coef = empirical_coefficients(args, modules)
    candidates = sample_candidates(modules, coef, count=args.count, n=args.candidates, seed=args.seed)
    if args.mode == "balanced":
        pairs = select_balanced_pairs(
            candidates,
            pairs=args.pairs,
            raw_window_frac=args.raw_window_frac,
            max_neighbors=args.max_neighbors,
            target_score_delta=args.target_score_delta,
        )
    else:
        pairs = select_pairs(candidates, pairs=args.pairs, raw_window_frac=args.raw_window_frac, max_neighbors=args.max_neighbors)
    policy_rows, pair_rows = serialize_pairs(pairs, args.count)
    out = args.output_root / "structural_scenarios"
    prefix = "sparse_nvfp4_empirical_balanced_scenario" if args.mode == "balanced" else "sparse_nvfp4_empirical_scenario"
    write_csv(out / f"{prefix}_policies.csv", policy_rows)
    write_csv(out / f"{prefix}_pairs.csv", pair_rows)
    write_report(out / f"{prefix}_summary.md", coef, pair_rows, args)
    plot_pairs(pair_rows, out / f"{prefix}_pairs.png")
    print(f"wrote {out / f'{prefix}_summary.md'}")


def load_modules(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.source_015_root / "sensitivity" / "module_method_kernel_local_errors.csv"
    rows = [row for row in read_csv(path) if row.get("method") == METHOD]
    return [
        {"name": row["module_name"], "layer": int(f(row, "layer")), "type": row["module_type"], "error": f(row, args.metric)}
        for row in rows
    ]


def empirical_coefficients(args: argparse.Namespace, modules: list[dict[str, Any]]) -> dict[str, float]:
    by_name = {row["name"]: row for row in modules}
    policies = {row["policy_id"]: row for row in read_csv(args.output_root / "stratified" / "policies" / "stratified_policies_sparse_nvfp4.csv")}
    losses = read_csv(args.output_root / "loss" / "loss_samples_sparse_nvfp4_stratified.csv")
    labels = ["count", "raw", *TYPES, *(name for _, _, name in BUCKETS)]
    x_rows = []
    y = []
    for loss in losses:
        policy = policies[loss["policy_id"]]
        features = {label: 0.0 for label in labels}
        features["count"] = f(loss, "selected_modules")
        for name in selected_from_text(policy["selected_names"]):
            module = by_name[name]
            error = module["error"]
            features["raw"] += error
            features[module["type"]] += error
            for lo, hi, bucket in BUCKETS:
                if lo <= module["layer"] <= hi:
                    features[bucket] += error
        x_rows.append([features[label] for label in labels])
        y.append(f(loss, "loss_delta_vs_dense"))
    x = np.array(x_rows, dtype=np.float64)
    yv = np.array(y, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    z = np.c_[np.ones(len(x)), (x - mu) / sd]
    penalty = np.eye(z.shape[1], dtype=np.float64) * 1e-3
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ yv)
    out = {"bias": float(coef[0])}
    for label, value, scale, center in zip(labels, coef[1:], sd, mu):
        out[f"{label}_standardized"] = float(value)
        out[f"{label}_raw"] = float(value / scale)
        out[f"{label}_mean"] = float(center)
    return out


def sample_candidates(modules: list[dict[str, Any]], coef: dict[str, float], *, count: int, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: dict[str, dict[str, Any]] = {}
    for _ in range(n):
        selected = rng.sample(modules, count)
        names = sorted(row["name"] for row in selected)
        key = selected_to_text(names)
        if key in out:
            continue
        raw = sum(row["error"] for row in selected)
        type_score = sum(row["error"] * coef[f"{row['type']}_raw"] for row in selected)
        depth_score = sum(empirical_depth_score(row, coef) for row in selected)
        score = type_score + depth_score
        out[key] = {
            "names": names,
            "raw": raw,
            "score": score,
            "type_score": type_score,
            "depth_score": depth_score,
            "composition": composition(selected),
        }
    return sorted(out.values(), key=lambda item: item["raw"])


def empirical_module_score(module: dict[str, Any], coef: dict[str, float]) -> float:
    score = module["error"] * coef[f"{module['type']}_raw"]
    score += empirical_depth_score(module, coef)
    return score


def empirical_depth_score(module: dict[str, Any], coef: dict[str, float]) -> float:
    for lo, hi, bucket in BUCKETS:
        if lo <= module["layer"] <= hi:
            return module["error"] * coef[f"{bucket}_raw"]
    return 0.0


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
    return ";".join(
        [
            ",".join(f"{key}:{value}" for key, value in layers.items()),
            ",".join(f"{key}:{types[key]}" for key in sorted(types)),
        ]
    )


def select_pairs(candidates: list[dict[str, Any]], *, pairs: int, raw_window_frac: float, max_neighbors: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    raw_values = [item["raw"] for item in candidates]
    window = (max(raw_values) - min(raw_values)) * raw_window_frac
    scored = []
    for i, left in enumerate(candidates):
        j = i + 1
        checked = 0
        while j < len(candidates) and candidates[j]["raw"] - left["raw"] <= window and checked < max_neighbors:
            right = candidates[j]
            low, high = (left, right) if left["score"] <= right["score"] else (right, left)
            raw_rel = abs(high["raw"] - low["raw"]) / max((high["raw"] + low["raw"]) / 2.0, 1e-12)
            score_gap = high["score"] - low["score"]
            scored.append((score_gap - raw_rel, low, high))
            j += 1
            checked += 1
    selected = []
    used = set()
    for _, low, high in sorted(scored, key=lambda item: item[0], reverse=True):
        low_key = selected_to_text(low["names"])
        high_key = selected_to_text(high["names"])
        if low_key in used or high_key in used:
            continue
        selected.append((low, high))
        used.update([low_key, high_key])
        if len(selected) == pairs:
            break
    return selected


def select_balanced_pairs(
    candidates: list[dict[str, Any]],
    *,
    pairs: int,
    raw_window_frac: float,
    max_neighbors: int,
    target_score_delta: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    raw_values = [item["raw"] for item in candidates]
    window = (max(raw_values) - min(raw_values)) * raw_window_frac
    scored = []
    for i, left in enumerate(candidates):
        j = i + 1
        checked = 0
        while j < len(candidates) and candidates[j]["raw"] - left["raw"] <= window and checked < max_neighbors:
            right = candidates[j]
            low, high = (left, right) if left["score"] <= right["score"] else (right, left)
            raw_rel = abs(high["raw"] - low["raw"]) / max((high["raw"] + low["raw"]) / 2.0, 1e-12)
            score_delta = high["score"] - low["score"]
            type_delta = high["type_score"] - low["type_score"]
            depth_delta = high["depth_score"] - low["depth_score"]
            if type_delta <= 0 or depth_delta <= 0:
                j += 1
                checked += 1
                continue
            balance = min(type_delta, depth_delta) / max(type_delta, depth_delta)
            objective = abs(score_delta - target_score_delta) + 2.0 * raw_rel - 0.03 * balance
            scored.append((objective, low, high))
            j += 1
            checked += 1
    selected = []
    used = set()
    for _, low, high in sorted(scored, key=lambda item: item[0]):
        low_key = selected_to_text(low["names"])
        high_key = selected_to_text(high["names"])
        if low_key in used or high_key in used:
            continue
        selected.append((low, high))
        used.update([low_key, high_key])
        if len(selected) == pairs:
            break
    return selected


def serialize_pairs(pairs: list[tuple[dict[str, Any], dict[str, Any]]], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies = []
    pair_rows = []
    for idx, (low, high) in enumerate(pairs):
        pair_id = f"sparse_nvfp4_empirical_c{count:03d}_pair{idx:02d}"
        for arm, item in (("low_empirical", low), ("high_empirical", high)):
            policies.append(
                {
                    "policy_id": f"{pair_id}_{arm}",
                    "pair_id": pair_id,
                    "arm": arm,
                    "sample_kind": "empirical_structural_raw_matched",
                    "method": METHOD,
                    "selected_modules": count,
                    "selected_names": selected_to_text(item["names"]),
                    "raw_error_sum": item["raw"],
                    "empirical_score_sum": item["score"],
                    "composition": item["composition"],
                }
            )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "selected_modules": count,
                "raw_rel_gap": abs(high["raw"] - low["raw"]) / max((high["raw"] + low["raw"]) / 2.0, 1e-12),
                "empirical_score_delta": high["score"] - low["score"],
                "type_score_delta": high["type_score"] - low["type_score"],
                "depth_score_delta": high["depth_score"] - low["depth_score"],
                "low_composition": low["composition"],
                "high_composition": high["composition"],
            }
        )
    return policies, pair_rows


def write_report(path: Path, coef: dict[str, float], pair_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Sparse NVFP4 Empirical Structural Scenario Summary",
        "",
        f"Count: `{args.count}`; pairs: `{len(pair_rows)}`.",
        "",
        "## Empirical Standardized Coefficients",
        "",
        "| feature | coef |",
        "|---|---:|",
    ]
    for key in ("count", "raw", *TYPES, *(name for _, _, name in BUCKETS)):
        lines.append(f"| {key} | {coef[f'{key}_standardized']:.6f} |")
    lines.extend(
        [
            "",
            "## Pair Quality",
            "",
            f"- Raw relative gap mean/max: `{mean(f(row, 'raw_rel_gap') for row in pair_rows):.6f}` / `{max(f(row, 'raw_rel_gap') for row in pair_rows):.6f}`",
            f"- Empirical score delta mean/max: `{mean(f(row, 'empirical_score_delta') for row in pair_rows):.6f}` / `{max(f(row, 'empirical_score_delta') for row in pair_rows):.6f}`",
            f"- Type score delta mean: `{mean(f(row, 'type_score_delta') for row in pair_rows):.6f}`",
            f"- Depth score delta mean: `{mean(f(row, 'depth_score_delta') for row in pair_rows):.6f}`",
            "",
            "## Plot",
            "",
            f"- `{path.parent / 'sparse_nvfp4_empirical_scenario_pairs.png'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_pairs(pair_rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([f(row, "raw_rel_gap") for row in pair_rows], [f(row, "empirical_score_delta") for row in pair_rows], alpha=0.85)
    ax.set_xlabel("Raw local relative gap")
    ax.set_ylabel("Empirical structural score delta")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
