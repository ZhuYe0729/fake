#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_pareto import DEBUG_ROOT, SCENARIO, f, read_csv, write_csv, write_json


METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pseudo-Pareto roots for Llama2 single-method E2E baselines.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.output_root / "costs" / "module_method_candidates.csv")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    for method in methods:
        root = args.output_root / "baselines" / method
        selected = [row for row in rows if row["method"] == method and str(row.get("supported", "True")).lower() == "true"]
        dense_modules = {row["module_name"] for row in rows if row["method"] == "dense_bf16"}
        if len(selected) != len(dense_modules):
            raise RuntimeError(f"{method} only has {len(selected)} supported modules, expected {len(dense_modules)}")
        summary = summarize(method, selected)
        write_policy(root, method, selected, summary)
        write_csv(root / "pareto" / "pareto_points.csv", [summary])
        print(f"wrote baseline {method}: modules={len(selected)} latency={summary['latency_ms']:.4f}")


def summarize(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "point_index": 0,
        "quality_budget": sum(f(row, "quality_cost") for row in rows),
        "quality_cost": sum(f(row, "quality_cost") for row in rows),
        "latency_ms": sum(f(row, "latency_cost") for row in rows),
        "total_prefill_ms": sum(f(row, "prefill_ms") for row in rows),
        "total_decode_ms": sum(f(row, "decode_ms") * SCENARIO["output_tokens"] for row in rows),
        "total_conversion_ms": sum(f(row, "conversion_ms") for row in rows),
        "selected_method": method,
    }


def write_policy(root: Path, method: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    modules = []
    for row in sorted(rows, key=lambda r: int(f(r, "module_index"))):
        if method == "dense_nvfp4_prefill_marlin_decode":
            prefill_backend = "dense_nvfp4"
            decode_backend = "marlin_nvfp4"
        else:
            prefill_backend = method
            decode_backend = method
        modules.append(
            {
                "name": row["linear_group"],
                "module_name": row["module_name"],
                "n": int(f(row, "out_features")),
                "k": int(f(row, "in_features")),
                "count": 1,
                "selected_prefill_backend": prefill_backend,
                "selected_decode_backend": decode_backend,
                "selected_total_ms": f(row, "latency_cost"),
                "selected_prefill_ms": f(row, "prefill_ms"),
                "selected_decode_ms": f(row, "decode_ms"),
                "selected_conversion_ms": f(row, "conversion_ms"),
                "quality_cost": f(row, "quality_cost"),
                "reason": f"single_method_baseline_{method}",
            }
        )
    write_json(
        root / "pareto" / "policies" / "point_000_budget_0.json",
        {
            "policy_format": "single_method_baseline_normal_02_v1",
            "scenario": SCENARIO,
            "summary": summary,
            "modules": modules,
        },
    )


if __name__ == "__main__":
    main()
