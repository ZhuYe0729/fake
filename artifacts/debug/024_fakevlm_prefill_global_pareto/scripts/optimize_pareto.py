#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common_fakevlm_pareto import DEBUG_ROOT, METHODS, f, pareto_policy_path, parse_batches, policy_counts, read_csv, write_csv, write_json, write_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize FakeVLM quality-constrained Pareto policies per batch.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--batches", default="all")
    parser.add_argument("--budgets", default="auto")
    parser.add_argument("--budget-bins", type=int, default=2000)
    parser.add_argument("--max-auto-points", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = {}
    for batch in parse_batches(args.batches):
        candidates = read_csv(args.output_root / "costs" / f"batch_{batch}" / "module_method_candidates.csv")
        modules = group_candidates(candidates)
        validate_modules(modules)
        max_quality = sum(max(f(row, "quality_cost") for row in rows) for rows in modules.values())
        scale = args.budget_bins / max_quality if max_quality > 0 else 1.0
        max_bins = sum(max(q_bin(row, scale) for row in rows) for rows in modules.values())
        budgets = parse_budgets(args.budgets, max_quality, max_bins, scale, args.max_auto_points)
        points = []
        previous = None
        for index, (budget, budget_int) in enumerate(budgets):
            policy = solve_budget(modules, budget_int, scale)
            point = summarize_policy(index, budget, policy, modules, batch)
            points.append(point)
            write_one_policy(args.output_root, batch, index, budget, policy, point)
            if previous is not None:
                write_diff(args.output_root, batch, index, previous, policy)
            previous = policy
        unique = dedupe(points)
        write_csv(args.output_root / "pareto" / f"batch_{batch}" / "pareto_points.csv", points)
        write_csv(args.output_root / "pareto" / f"batch_{batch}" / "pareto_unique_points.csv", unique)
        metadata[batch] = {"points": len(points), "unique_points": len(unique), "modules": len(modules), "max_quality": max_quality}
        print(f"batch={batch}: wrote {len(points)} points ({len(unique)} unique)")
    write_json(args.output_root / "pareto" / "optimize_metadata.json", metadata)


def group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["module_name"], []).append(row)
    for items in out.values():
        items.sort(key=lambda row: METHODS.index(row["method"]))
    return dict(sorted(out.items(), key=lambda item: int(f(item[1][0], "module_index"))))


def validate_modules(modules: dict[str, list[dict[str, Any]]]) -> None:
    expected = set(METHODS)
    bad = [(name, sorted({row["method"] for row in rows})) for name, rows in modules.items() if {row["method"] for row in rows} != expected]
    if bad:
        raise RuntimeError(f"incomplete candidate sets: {len(bad)} sample={bad[:3]}")


def parse_budgets(spec: str, max_quality: float, max_bins: int, scale: float, max_points: int) -> list[tuple[float, int]]:
    if spec != "auto":
        values = [float(item) for item in spec.split(",") if item.strip()]
        if values and max(values) <= 1.0:
            return [(value * max_quality, int(round(value * max_bins))) for value in values]
        return [(value, int(math.ceil(value * scale))) for value in values]
    if max_quality <= 0:
        return [(0.0, 0)]
    ratios = [0.0]
    ratios.extend(10 ** (-3 + i * (3 / max(max_points - 2, 1))) for i in range(max_points - 1))
    ratios[-1] = 1.0
    return sorted({(ratio * max_quality, int(round(ratio * max_bins))) for ratio in ratios})


def solve_budget(modules: dict[str, list[dict[str, Any]]], budget: int, scale: float) -> dict[str, dict[str, Any]]:
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    items = list(modules.items())
    for _name, rows in items:
        nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (lat, choices) in dp.items():
            for idx, row in enumerate(rows):
                q = q_bin(row, scale)
                new_q = used + q
                if new_q > budget:
                    continue
                new_lat = lat + f(row, "latency_cost")
                old = nxt.get(new_q)
                if old is None or new_lat < old[0]:
                    nxt[new_q] = (new_lat, choices + (idx,))
        if not nxt:
            raise RuntimeError(f"no feasible policy at budget={budget}")
        dp = prune(nxt)
    _q, (lat, choices) = min(dp.items(), key=lambda item: (item[1][0], item[0]))
    return {name: dict(rows[idx], _total_latency=lat) for (name, rows), idx in zip(items, choices)}


def q_bin(row: dict[str, Any], scale: float) -> int:
    quality = f(row, "quality_cost")
    return 0 if quality <= 0 else max(1, int(math.ceil(quality * scale)))


def prune(states: dict[int, tuple[float, tuple[int, ...]]]) -> dict[int, tuple[float, tuple[int, ...]]]:
    out = {}
    best = math.inf
    for q in sorted(states):
        lat, choices = states[q]
        if lat < best:
            out[q] = (lat, choices)
            best = lat
    return out


def summarize_policy(index: int, budget: float, policy: dict[str, dict[str, Any]], modules: dict[str, list[dict[str, Any]]], batch: int) -> dict[str, Any]:
    selected = list(policy.values())
    dense_latency = sum(f(next(row for row in rows if row["method"] == "dense_bf16"), "latency_cost") for rows in modules.values())
    latency = sum(f(row, "latency_cost") for row in selected)
    quality = sum(f(row, "quality_cost") for row in selected)
    counts = policy_counts([{"selected_method": row["method"]} for row in selected])
    return {
        "batch_size": batch,
        "point_index": index,
        "quality_budget": budget,
        "quality_cost": quality,
        "latency_ms": latency,
        "dense_latency_ms": dense_latency,
        "speedup_vs_dense_linear": dense_latency / latency if latency > 0 else 0.0,
        **{f"count_{method}": counts.get(method, 0) for method in METHODS},
    }


def write_one_policy(output_root: Path, batch: int, index: int, budget: float, policy: dict[str, dict[str, Any]], point: dict[str, Any]) -> None:
    modules = []
    for row in policy.values():
        modules.append(
            {
                "name": row["module_name"],
                "module_name": row["module_name"],
                "module_index": int(f(row, "module_index")),
                "layer": int(f(row, "layer")),
                "module_type": row["module_type"],
                "module_family": row["module_family"],
                "n": int(f(row, "out_features")),
                "k": int(f(row, "in_features")),
                "selected_method": row["method"],
                "backend": row["method"],
                "quality_cost": f(row, "quality_cost"),
                "latency_cost": f(row, "latency_cost"),
            }
        )
    path = pareto_policy_path(output_root, batch, index, budget)
    write_policy(path, family=f"pareto_batch_{batch}", modules=modules, summary=point, scenario={"mode": "prefill_only", "batch_size": batch})


def write_diff(output_root: Path, batch: int, index: int, old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]) -> None:
    rows = []
    for name in old:
        if old[name]["method"] == new[name]["method"]:
            continue
        rows.append(
            {
                "module_name": name,
                "from_method": old[name]["method"],
                "to_method": new[name]["method"],
                "latency_delta_ms": f(new[name], "latency_cost") - f(old[name], "latency_cost"),
                "quality_delta": f(new[name], "quality_cost") - f(old[name], "quality_cost"),
            }
        )
    write_csv(output_root / "pareto" / f"batch_{batch}" / "diffs" / f"diff_to_point_{index:03d}.csv", rows)


def dedupe(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in points:
        key = (round(f(row, "quality_cost"), 12), round(f(row, "latency_ms"), 8), tuple(int(row[f"count_{m}"]) for m in METHODS))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


if __name__ == "__main__":
    main()
