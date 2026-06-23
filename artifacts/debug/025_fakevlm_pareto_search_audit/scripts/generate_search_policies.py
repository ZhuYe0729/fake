#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from common_search_audit import (
    DEBUG_ROOT,
    DEFAULT_BATCH_SIZE,
    METHODS,
    SOURCE_024_ROOT,
    cost_rows,
    f,
    read_json,
    selected_024_rows,
    write_csv,
    write_json,
)

from common_fakevlm_pareto import write_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FakeVLM search-audit policies.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--random-policies", type=int, default=30)
    parser.add_argument("--neighborhood-policies", type=int, default=20)
    parser.add_argument("--suspicious-policies", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = cost_rows(args.batch_size)
    modules = module_order(rows)
    by_module_method = {(row["module_name"], row["method"]): row for row in rows}
    dense_latency = sum(f(by_module_method[(name, "dense_bf16")], "latency_cost") for name in modules)
    policies: list[dict[str, Any]] = []

    for index, ratio in enumerate(random_ratios(args.random_policies), start=1):
        methods = []
        for _name in modules:
            method = "dense_bf16" if rng.random() > ratio else rng.choice(METHODS[1:])
            methods.append(method)
        policies.append(write_search_policy(args, by_module_method, modules, methods, "random", f"random_{index:03d}", {"replacement_ratio": ratio}, dense_latency))

    parents = representative_parents(args.batch_size)
    mutation_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
    neighborhood_written = 0
    for parent in parents:
        parent_methods = methods_from_policy(Path(parent["policy_json"]))
        for rate in mutation_rates:
            if neighborhood_written >= args.neighborhood_policies:
                break
            methods = mutate_methods(parent_methods, rate, rng)
            neighborhood_written += 1
            policies.append(
                write_search_policy(
                    args,
                    by_module_method,
                    modules,
                    methods,
                    "neighborhood",
                    f"neighborhood_{neighborhood_written:03d}",
                    {"parent_point": int(f(parent, "point_index")), "mutation_rate": rate},
                    dense_latency,
                )
            )
        if neighborhood_written >= args.neighborhood_policies:
            break

    suspicious_modules = select_suspicious_modules(rows, modules, limit=32)
    suspicious_written = 0
    suspicious_parents = parents[-2:] if len(parents) >= 2 else parents
    for parent in suspicious_parents:
        parent_methods = methods_from_policy(Path(parent["policy_json"]))
        for _ in range(max(1, args.suspicious_policies // max(len(suspicious_parents), 1))):
            if suspicious_written >= args.suspicious_policies:
                break
            methods = list(parent_methods)
            for name in suspicious_modules:
                idx = modules.index(name)
                methods[idx] = rng.choice(METHODS)
            suspicious_written += 1
            policies.append(
                write_search_policy(
                    args,
                    by_module_method,
                    modules,
                    methods,
                    "suspicious",
                    f"suspicious_{suspicious_written:03d}",
                    {
                        "parent_point": int(f(parent, "point_index")),
                        "suspicious_module_count": len(suspicious_modules),
                    },
                    dense_latency,
                )
            )

    write_csv(args.output_root / "search" / "search_policies.csv", policies)
    write_json(
        args.output_root / "search" / "search_policies_metadata.json",
        {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "policy_count": len(policies),
            "methods": METHODS,
            "source_024_root": SOURCE_024_ROOT,
            "suspicious_modules": suspicious_modules,
        },
    )
    print(f"wrote search policies={len(policies)}")


def module_order(rows: list[dict[str, Any]]) -> list[str]:
    names = sorted({row["module_name"] for row in rows}, key=lambda name: int(f(next(row for row in rows if row["module_name"] == name), "module_index")))
    return names


def random_ratios(count: int) -> list[float]:
    base = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
    return [base[i % len(base)] for i in range(count)]


def representative_parents(batch_size: int) -> list[dict[str, Any]]:
    rows = selected_024_rows(batch_size)
    wanted = [0, 13, 21, 25]
    by_point = {int(f(row, "point_index")): row for row in rows}
    selected = [by_point[point] for point in wanted if point in by_point]
    if len(selected) == len(wanted):
        return selected
    rows = sorted(rows, key=lambda row: int(f(row, "point_index")))
    if len(rows) <= 4:
        return rows
    indices = sorted({round(index * (len(rows) - 1) / 3) for index in range(4)})
    return [rows[index] for index in indices]


def methods_from_policy(path: Path) -> list[str]:
    payload = read_json(path)
    return [str(item.get("selected_method") or item.get("backend")) for item in payload["modules"]]


def mutate_methods(methods: list[str], rate: float, rng: random.Random) -> list[str]:
    out = list(methods)
    count = max(1, round(len(out) * rate))
    for idx in rng.sample(range(len(out)), count):
        choices = [method for method in METHODS if method != out[idx]]
        out[idx] = rng.choice(choices)
    return out


def select_suspicious_modules(rows: list[dict[str, Any]], modules: list[str], *, limit: int) -> list[str]:
    by_module: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_module.setdefault(row["module_name"], []).append(row)
    scored = []
    for name in modules:
        dense = next(row for row in by_module[name] if row["method"] == "dense_bf16")
        candidates = [row for row in by_module[name] if row["method"] != "dense_bf16"]
        best_gain = max(f(dense, "latency_cost") - f(row, "latency_cost") for row in candidates)
        min_quality = min(f(row, "quality_cost") for row in candidates)
        max_quality = max(f(row, "quality_cost") for row in candidates)
        score = best_gain / max(min_quality, 1e-9) + 0.1 * max_quality
        scored.append((score, name))
    scored.sort(reverse=True)
    return [name for _score, name in scored[:limit]]


def policy_module(row: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "name": row["module_name"],
        "module_name": row["module_name"],
        "module_index": int(f(row, "module_index")),
        "layer": int(f(row, "layer")),
        "module_type": row.get("module_type", ""),
        "module_family": row.get("module_family", ""),
        "n": int(f(row, "out_features")),
        "k": int(f(row, "in_features")),
        "selected_method": method,
        "backend": method,
        "quality_cost": f(row, "quality_cost"),
        "latency_cost": f(row, "latency_cost"),
    }


def write_search_policy(
    args: argparse.Namespace,
    by_module_method: dict[tuple[str, str], dict[str, Any]],
    modules: list[str],
    methods: list[str],
    family: str,
    name: str,
    extra: dict[str, Any],
    dense_latency: float,
) -> dict[str, Any]:
    policy_modules = [policy_module(by_module_method[(module_name, method)], method) for module_name, method in zip(modules, methods)]
    counts = Counter(methods)
    latency = sum(item["latency_cost"] for item in policy_modules)
    quality = sum(item["quality_cost"] for item in policy_modules)
    key = f"{family}_{name}"
    path = args.output_root / "policies" / family / f"{name}.json"
    summary = {
        "key": key,
        "family": family,
        "batch_size": args.batch_size,
        "predicted_linear_latency_ms": latency,
        "predicted_linear_speedup_vs_dense": dense_latency / latency if latency > 0 else 0.0,
        "predicted_quality_cost": quality,
        "backend_counts": dict(sorted(counts.items())),
        **extra,
    }
    write_policy(path, family=family, modules=policy_modules, summary=summary, scenario={"mode": "prefill_only", "batch_size": args.batch_size})
    return {
        "key": key,
        "family": family,
        "policy_name": name,
        "policy_json": str(path),
        **summary,
        **{f"count_{method}": counts.get(method, 0) for method in METHODS},
    }


if __name__ == "__main__":
    main()
