#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Any

from common_pareto import DEBUG_ROOT, METHODS, f, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Pareto quality-speed outputs.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_csv(args.output_root / "costs" / "module_method_candidates.csv")
    points = read_csv(args.output_root / "pareto" / "pareto_points.csv")
    unique = read_csv(args.output_root / "pareto" / "pareto_unique_points.csv")
    method_summary = summarize_methods(candidates)
    point_summary = summarize_points(unique)
    write_csv(args.output_root / "summary" / "method_cost_summary.csv", method_summary)
    write_csv(args.output_root / "summary" / "frontier_summary.csv", point_summary)
    write_analysis(args.output_root / "summary" / "analysis.md", method_summary, point_summary, len(candidates), len(points), len(unique))
    write_json(
        args.output_root / "summary" / "summary_metadata.json",
        {
            "candidate_rows": len(candidates),
            "pareto_points": len(points),
            "unique_pareto_points": len(unique),
            "outputs": ["method_cost_summary.csv", "frontier_summary.csv", "analysis.md"],
        },
    )
    print(f"summary written to {args.output_root / 'summary'}")


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        items = [row for row in rows if row.get("method") == method]
        if not items:
            continue
        lat = [f(row, "latency_cost") for row in items]
        q = [f(row, "quality_cost") for row in items]
        gains = [f(row, "latency_gain_vs_dense") for row in items]
        out.append(
            {
                "method": method,
                "rows": len(items),
                "latency_sum_ms": sum(lat),
                "quality_sum": sum(q),
                "latency_mean_ms": mean(lat),
                "quality_mean": mean(q),
                "latency_gain_sum_vs_dense": sum(gains),
                "latency_gain_mean_vs_dense": mean(gains),
            }
        )
    return out


def summarize_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    previous = None
    for row in rows:
        item = dict(row)
        if previous is None:
            item["delta_latency_vs_prev"] = ""
            item["delta_quality_vs_prev"] = ""
        else:
            item["delta_latency_vs_prev"] = f(row, "latency_ms") - f(previous, "latency_ms")
            item["delta_quality_vs_prev"] = f(row, "quality_cost") - f(previous, "quality_cost")
        out.append(item)
        previous = row
    return out


def write_analysis(
    path: Path,
    method_summary: list[dict[str, Any]],
    point_summary: list[dict[str, Any]],
    candidate_rows: int,
    points: int,
    unique_points: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_speed = min(point_summary, key=lambda row: f(row, "latency_ms")) if point_summary else None
    dense = next((row for row in point_summary if f(row, "quality_cost") == 0.0), point_summary[0] if point_summary else None)
    lines = [
        "# Llama2 Prefill-Only Pareto Analysis",
        "",
        "## Inputs",
        "",
        f"- Candidate rows: {candidate_rows}",
        f"- Pareto budget points: {points}",
        f"- Unique frontier points: {unique_points}",
        "",
        "## Method Cost Summary",
        "",
    ]
    for row in method_summary:
        lines.append(
            f"- {row['method']}: latency_sum_ms={float(row['latency_sum_ms']):.6f}, "
            f"quality_sum={float(row['quality_sum']):.6f}, "
            f"latency_gain_vs_dense={float(row['latency_gain_sum_vs_dense']):.6f}"
        )
    if dense and best_speed:
        lines.extend(
            [
                "",
                "## Frontier Endpoints",
                "",
                f"- Conservative endpoint: quality={float(dense['quality_cost']):.6f}, latency_ms={float(dense['latency_ms']):.6f}, speedup={float(dense['speedup_vs_dense_linear']):.6f}",
                f"- Speed endpoint: quality={float(best_speed['quality_cost']):.6f}, latency_ms={float(best_speed['latency_ms']):.6f}, speedup={float(best_speed['speedup_vs_dense_linear']):.6f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
