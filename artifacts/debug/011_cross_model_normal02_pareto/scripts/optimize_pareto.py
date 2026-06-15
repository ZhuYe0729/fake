#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common_pareto import (
    DEBUG_ROOT,
    METHODS,
    SCENARIO,
    f,
    policy_method_counts,
    read_csv,
    sanitize,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve quality-constrained per-linear Pareto policies for normal_02.")
    parser.add_argument("--model", choices=["llama2-7b", "llama31-8b", "qwen35-9b"], default="llama31-8b")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--budgets", default="auto")
    parser.add_argument("--budget-bins", type=int, default=2000)
    parser.add_argument("--max-auto-points", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root == DEBUG_ROOT and args.model != "llama2-7b":
        args.output_root = DEBUG_ROOT / "models" / args.model
    candidates = read_csv(args.output_root / "costs" / "module_method_candidates.csv")
    modules = group_candidates(candidates)
    validate_modules(modules)
    max_quality = sum(max(f(row, "quality_cost") for row in rows) for rows in modules.values())
    scale = args.budget_bins / max_quality if max_quality > 0 else 1.0
    max_quality_bins = sum(max(q_bin(row, scale) for row in rows) for rows in modules.values())
    budgets = parse_budgets(args.budgets, max_quality, max_quality_bins, scale, args.max_auto_points)
    points = []
    previous_policy: dict[str, dict[str, Any]] | None = None
    for index, (display_budget, budget_int) in enumerate(budgets):
        policy = solve_budget(modules, budget_int, scale)
        point = summarize_policy(index, display_budget, policy, modules)
        point["quality_budget_bins"] = budget_int
        points.append(point)
        write_policy(args.output_root, index, display_budget, policy, point)
        if previous_policy is not None:
            write_diff(args.output_root, index, previous_policy, policy)
        previous_policy = policy

    unique = dedupe_points(points)
    write_csv(args.output_root / "pareto" / "pareto_points.csv", points)
    write_csv(args.output_root / "pareto" / "pareto_unique_points.csv", unique)
    write_json(
        args.output_root / "pareto" / "optimize_metadata.json",
        {
            "modules": len(modules),
            "candidate_rows": len(candidates),
            "max_quality": max_quality,
            "max_quality_bins": max_quality_bins,
            "budget_bins": args.budget_bins,
            "scale": scale,
            "budget_points": len(budgets),
            "unique_points": len(unique),
            "scenario": SCENARIO,
        },
    )
    print(f"wrote {len(points)} pareto points ({len(unique)} unique)")


def group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    modules: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("supported", "True")).lower() != "true":
            continue
        modules.setdefault(str(row["module_name"]), []).append(row)
    for module_name, items in modules.items():
        items.sort(key=lambda row: METHODS.index(row["method"]) if row["method"] in METHODS else 999)
    return dict(sorted(modules.items(), key=lambda item: int(f(item[1][0], "module_index"))))


def validate_modules(modules: dict[str, list[dict[str, Any]]]) -> None:
    missing_dense = []
    for name, rows in modules.items():
        got = {row["method"] for row in rows}
        if "dense_bf16" not in got:
            missing_dense.append(name)
    if missing_dense:
        raise RuntimeError(f"modules missing dense_bf16: {len(missing_dense)}; {missing_dense[:5]}")


def parse_budgets(spec: str, max_quality: float, max_quality_bins: int, scale: float, max_points: int) -> list[tuple[float, int]]:
    if spec != "auto":
        values = [float(item) for item in spec.split(",") if item.strip()]
        if values and max(values) <= 1.0:
            return [(value * max_quality, int(round(value * max_quality_bins))) for value in values]
        return [(value, int(math.ceil(value * scale))) for value in values]
    ratios = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
    return sorted({(min(max(ratio, 0.0), 1.0) * max_quality, int(round(min(max(ratio, 0.0), 1.0) * max_quality_bins))) for ratio in ratios})


def solve_budget(modules: dict[str, list[dict[str, Any]]], budget_int: int, scale: float) -> dict[str, dict[str, Any]]:
    budget_int = max(0, budget_int)
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    module_items = list(modules.items())
    for _module_name, rows in module_items:
        next_dp: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used_q, (lat, choices) in dp.items():
            for choice_index, row in enumerate(rows):
                q = q_bin(row, scale)
                new_q = used_q + q
                if new_q > budget_int:
                    continue
                new_lat = lat + f(row, "latency_cost")
                previous = next_dp.get(new_q)
                if previous is None or new_lat < previous[0]:
                    next_dp[new_q] = (new_lat, choices + (choice_index,))
        if not next_dp:
            raise RuntimeError(f"no feasible state at budget={budget_int}")
        dp = prune_states(next_dp)
    best_q, (best_lat, best_choices) = min(dp.items(), key=lambda item: (item[1][0], item[0]))
    policy = {}
    for (module_name, rows), choice_index in zip(module_items, best_choices):
        selected = dict(rows[choice_index])
        selected["_quality_bin"] = best_q
        selected["_total_latency"] = best_lat
        policy[module_name] = selected
    return policy


def q_bin(row: dict[str, Any], scale: float) -> int:
    quality = f(row, "quality_cost")
    if quality <= 0.0:
        return 0
    return max(1, int(math.ceil(quality * scale)))


def prune_states(states: dict[int, tuple[float, tuple[int, ...]]]) -> dict[int, tuple[float, tuple[int, ...]]]:
    pruned: dict[int, tuple[float, tuple[int, ...]]] = {}
    best_latency = math.inf
    for q in sorted(states):
        latency, choices = states[q]
        if latency < best_latency:
            pruned[q] = (latency, choices)
            best_latency = latency
    return pruned


def summarize_policy(index: int, budget: float, policy: dict[str, dict[str, Any]], modules: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    selected_rows = list(policy.values())
    dense_latency = sum(f(rows_by_method(rows).get("dense_bf16", rows[0]), "latency_cost") for rows in modules.values())
    speed_latency = sum(min(f(row, "latency_cost") for row in rows) for rows in modules.values())
    total_latency = sum(f(row, "latency_cost") for row in selected_rows)
    total_quality = sum(f(row, "quality_cost") for row in selected_rows)
    total_prefill = sum(f(row, "prefill_ms") for row in selected_rows)
    total_decode = sum(f(row, "decode_ms") * SCENARIO["output_tokens"] for row in selected_rows)
    total_conversion = sum(f(row, "conversion_ms") for row in selected_rows)
    counts = policy_method_counts([{"selected_method": row["method"]} for row in selected_rows])
    return {
        "point_index": index,
        "quality_budget": budget,
        "quality_cost": total_quality,
        "latency_ms": total_latency,
        "speedup_vs_dense_linear": dense_latency / total_latency if total_latency > 0 else 0.0,
        "dense_latency_ms": dense_latency,
        "speed_optimal_latency_ms": speed_latency,
        "speed_gap_to_optimal_ms": total_latency - speed_latency,
        "total_prefill_ms": total_prefill,
        "total_decode_ms": total_decode,
        "total_conversion_ms": total_conversion,
        **{f"count_{method}": counts.get(method, 0) for method in METHODS},
    }


def rows_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in rows}


def write_policy(output_root: Path, index: int, budget: float, policy: dict[str, dict[str, Any]], point: dict[str, Any]) -> None:
    policy_rows = []
    modules_json = []
    for module_name, row in policy.items():
        item = {
            "module_name": module_name,
            "module_index": int(f(row, "module_index")),
            "layer": int(f(row, "layer")),
            "module_type": row["module_type"],
            "module_family": row["module_family"],
            "linear_group": row["linear_group"],
            "n": int(f(row, "out_features")),
            "k": int(f(row, "in_features")),
            "selected_method": row["method"],
            "quality_cost": f(row, "quality_cost"),
            "latency_cost": f(row, "latency_cost"),
            "prefill_ms": f(row, "prefill_ms"),
            "decode_ms": f(row, "decode_ms"),
            "conversion_ms": f(row, "conversion_ms"),
            "decode_backend": row.get("decode_backend", row["method"]),
            "output_tokens": SCENARIO["output_tokens"],
        }
        policy_rows.append(item)
        modules_json.append(
            {
                "name": row["linear_group"],
                "module_name": module_name,
                "n": int(f(row, "out_features")),
                "k": int(f(row, "in_features")),
                "count": 1,
                "selected_prefill_backend": (
                    "dense_nvfp4" if row["method"] == "dense_nvfp4_prefill_marlin_decode" else row["method"]
                ),
                "selected_decode_backend": row.get("decode_backend", row["method"]),
                "selected_total_ms": f(row, "latency_cost"),
                "selected_prefill_ms": f(row, "prefill_ms"),
                "selected_decode_ms": f(row, "decode_ms"),
                "selected_conversion_ms": f(row, "conversion_ms"),
                "quality_cost": f(row, "quality_cost"),
                "reason": "quality_constrained_pareto_normal_02",
            }
        )
    name = f"point_{index:03d}_budget_{sanitize(f'{budget:.6g}')}"
    write_csv(output_root / "pareto" / "policies" / f"{name}.csv", policy_rows)
    write_json(
        output_root / "pareto" / "policies" / f"{name}.json",
        {
            "policy_format": "quality_constrained_pareto_normal_02_v1",
            "scenario": SCENARIO,
            "summary": point,
            "modules": modules_json,
        },
    )


def write_diff(output_root: Path, index: int, old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]) -> None:
    rows = []
    for module_name in old:
        prev = old[module_name]
        curr = new[module_name]
        if prev["method"] == curr["method"]:
            continue
        rows.append(
            {
                "module_name": module_name,
                "layer": curr["layer"],
                "module_type": curr["module_type"],
                "module_family": curr["module_family"],
                "from_method": prev["method"],
                "to_method": curr["method"],
                "latency_delta_ms": f(curr, "latency_cost") - f(prev, "latency_cost"),
                "quality_delta": f(curr, "quality_cost") - f(prev, "quality_cost"),
            }
        )
    write_csv(output_root / "pareto" / "diffs" / f"diff_to_point_{index:03d}.csv", rows)


def dedupe_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen: set[tuple[float, float, tuple[int, ...]]] = set()
    for row in points:
        key = (
            round(float(row["quality_cost"]), 10),
            round(float(row["latency_ms"]), 10),
            tuple(int(row[f"count_{method}"]) for method in METHODS),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


if __name__ == "__main__":
    main()
