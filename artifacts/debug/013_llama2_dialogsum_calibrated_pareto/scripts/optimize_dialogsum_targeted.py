#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize targeted DialogSum-calibrated normal02 policies.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--budgets", default="0,0.01,0.02,0.035,0.05,0.065,0.08,0.10")
    parser.add_argument("--budget-bins", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.output_root / "costs" / "module_method_candidates.csv")
    modules = group_candidates(rows)
    validate_modules(modules)
    max_quality = max(sum(f(row, "quality_cost") for row in select_uniform(modules, method)) for method in METHODS)
    scale = args.budget_bins / max_quality if max_quality > 0 else 1.0
    budgets = parse_budgets(args.budgets, max_quality, scale)
    points = []
    previous_policy = None
    for index, (budget, budget_int) in enumerate(budgets):
        policy = solve_budget(modules, budget_int, scale)
        point = summarize_policy(index, budget, policy, modules)
        point["quality_budget_bins"] = budget_int
        points.append(point)
        write_policy(args.output_root, index, budget, policy, point)
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
            "candidate_rows": len(rows),
            "max_quality": max_quality,
            "scale": scale,
            "budgets": budgets,
            "points": len(points),
            "unique_points": len(unique),
            "scenario": SCENARIO,
        },
    )
    print(f"wrote {len(points)} targeted points ({len(unique)} unique)")


def group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    modules: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("supported", "True")).lower() != "true":
            continue
        modules.setdefault(row["module_name"], []).append(row)
    for items in modules.values():
        items.sort(key=lambda row: METHODS.index(row["method"]) if row["method"] in METHODS else 999)
    return dict(sorted(modules.items(), key=lambda item: int(f(item[1][0], "module_index"))))


def validate_modules(modules: dict[str, list[dict[str, Any]]]) -> None:
    missing = [name for name, rows in modules.items() if "dense_bf16" not in {row["method"] for row in rows}]
    if missing:
        raise RuntimeError(f"modules missing dense_bf16: {missing[:5]}")


def select_uniform(modules: dict[str, list[dict[str, Any]]], method: str) -> list[dict[str, Any]]:
    rows = []
    for choices in modules.values():
        by_method = {row["method"]: row for row in choices}
        if method in by_method:
            rows.append(by_method[method])
    return rows


def parse_budgets(spec: str, max_quality: float, scale: float) -> list[tuple[float, int]]:
    values = [float(item) for item in spec.split(",") if item.strip()]
    return [(value, int(math.ceil(value * scale))) for value in values if 0 <= value <= max_quality + 1e-12]


def solve_budget(modules: dict[str, list[dict[str, Any]]], budget_int: int, scale: float) -> dict[str, dict[str, Any]]:
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    module_items = list(modules.items())
    for _name, rows in module_items:
        next_dp: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used_q, (lat, choices) in dp.items():
            for choice_index, row in enumerate(rows):
                new_q = used_q + q_bin(row, scale)
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
    if quality <= 0:
        return 0
    return max(1, int(math.ceil(quality * scale)))


def prune_states(states: dict[int, tuple[float, tuple[int, ...]]]) -> dict[int, tuple[float, tuple[int, ...]]]:
    pruned = {}
    best_latency = math.inf
    for q in sorted(states):
        latency, choices = states[q]
        if latency < best_latency:
            pruned[q] = (latency, choices)
            best_latency = latency
    return pruned


def summarize_policy(index: int, budget: float, policy: dict[str, dict[str, Any]], modules: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    selected = list(policy.values())
    dense_latency = sum(f({row["method"]: row for row in rows}.get("dense_bf16", rows[0]), "latency_cost") for rows in modules.values())
    total_latency = sum(f(row, "latency_cost") for row in selected)
    counts = method_counts(selected)
    return {
        "point_index": index,
        "quality_budget": budget,
        "quality_cost": sum(f(row, "quality_cost") for row in selected),
        "original_quality_cost": sum(f(row, "original_quality_cost") for row in selected),
        "latency_ms": total_latency,
        "speedup_vs_dense_linear": dense_latency / total_latency if total_latency > 0 else 0.0,
        "dense_latency_ms": dense_latency,
        "total_prefill_ms": sum(f(row, "prefill_ms") for row in selected),
        "total_decode_ms": sum(f(row, "decode_ms") * SCENARIO["output_tokens"] for row in selected),
        "total_conversion_ms": sum(f(row, "conversion_ms") for row in selected),
        **{f"count_{method}": counts.get(method, 0) for method in METHODS},
    }


def method_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["method"]] = counts.get(row["method"], 0) + 1
    return counts


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
            "original_quality_cost": f(row, "original_quality_cost"),
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
                "selected_prefill_backend": "dense_nvfp4" if row["method"] == "dense_nvfp4_prefill_marlin_decode" else row["method"],
                "selected_decode_backend": row.get("decode_backend", row["method"]),
                "selected_total_ms": f(row, "latency_cost"),
                "selected_prefill_ms": f(row, "prefill_ms"),
                "selected_decode_ms": f(row, "decode_ms"),
                "selected_conversion_ms": f(row, "conversion_ms"),
                "quality_cost": f(row, "quality_cost"),
                "original_quality_cost": f(row, "original_quality_cost"),
                "reason": "dialogsum_calibrated_targeted_normal_02",
            }
        )
    name = f"point_{index:03d}_budget_{sanitize(f'{budget:.6g}')}"
    write_csv(output_root / "pareto" / "policies" / f"{name}.csv", policy_rows)
    write_json(
        output_root / "pareto" / "policies" / f"{name}.json",
        {
            "policy_format": "dialogsum_calibrated_targeted_normal_02_v1",
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
    seen = set()
    for row in points:
        key = (
            round(float(row["quality_cost"]), 12),
            round(float(row["latency_ms"]), 8),
            tuple(int(row[f"count_{method}"]) for method in METHODS),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def sanitize(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


if __name__ == "__main__":
    main()
