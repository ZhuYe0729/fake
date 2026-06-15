#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from common_pareto import DEBUG_ROOT, METHODS, SCENARIO, f, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Pareto quality-speed outputs for normal_02.")
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
    write_analysis(
        args.output_root / "summary" / "analysis.md",
        method_summary,
        point_summary,
        len(candidates),
        len(points),
        len(unique),
    )
    write_json(
        args.output_root / "summary" / "summary_metadata.json",
        {
            "candidate_rows": len(candidates),
            "pareto_points": len(points),
            "unique_pareto_points": len(unique),
            "scenario": SCENARIO,
            "outputs": ["method_cost_summary.csv", "frontier_summary.csv", "analysis.md"],
        },
    )
    print(f"summary written to {args.output_root / 'summary'}")


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        items = [row for row in rows if row.get("method") == method and str(row.get("supported", "True")).lower() == "true"]
        if not items:
            items_unsupported = [row for row in rows if row.get("method") == method]
            if items_unsupported:
                out.append(
                    {
                        "method": method,
                        "rows": len(items_unsupported),
                        "supported_rows": 0,
                        "latency_sum_ms": "",
                        "quality_sum": "",
                        "note": "all rows unsupported",
                    }
                )
            continue
        lat = [f(row, "latency_cost") for row in items]
        q = [f(row, "quality_cost") for row in items]
        gains = [f(row, "latency_gain_vs_dense") for row in items]
        prefill = [f(row, "prefill_ms") for row in items]
        decode = [f(row, "decode_ms") for row in items]
        conv = [f(row, "conversion_ms") for row in items]
        out.append(
            {
                "method": method,
                "rows": len(items),
                "supported_rows": len(items),
                "latency_sum_ms": sum(lat),
                "quality_sum": sum(q),
                "latency_mean_ms": mean(lat),
                "quality_mean": mean(q),
                "latency_gain_sum_vs_dense": sum(gains),
                "latency_gain_mean_vs_dense": mean(gains),
                "prefill_sum_ms": sum(prefill),
                "decode_sum_ms": sum(decode),
                "conversion_sum_ms": sum(conv),
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
        total_ms = f(row, "latency_ms")
        item["decode_fraction"] = f(row, "total_decode_ms") / total_ms if total_ms > 0 else 0.0
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
        "# Llama2 Normal-02 Pareto Analysis",
        "",
        f"Scenario: batch_size={SCENARIO['batch_size']}, input_tokens={SCENARIO['input_tokens']}, "
        f"output_tokens={SCENARIO['output_tokens']} (prefill M={SCENARIO['m_prefill']}, decode M={SCENARIO['m_decode']})",
        "",
        "## Inputs",
        "",
        f"- Candidate rows: {candidate_rows}",
        f"- Pareto budget points: {points}",
        f"- Unique frontier points: {unique_points}",
        "",
        "## Method Cost Summary",
        "",
        "| Method | rows | latency_sum_ms | quality_sum | prefill_sum_ms | decode_sum_ms | conv_sum_ms | gain_vs_dense |",
        "|--------|------|---------------|-------------|----------------|---------------|-------------|---------------|",
    ]
    for row in method_summary:
        if row.get("note"):
            lines.append(f"| {row['method']} | {row['rows']} | - | - | - | - | - | {row['note']} |")
            continue
        lines.append(
            f"| {row['method']} | {row['rows']} | {float(row['latency_sum_ms']):.2f} | {float(row['quality_sum']):.4f} | "
            f"{float(row['prefill_sum_ms']):.2f} | {float(row['decode_sum_ms']):.4f} | "
            f"{float(row['conversion_sum_ms']):.4f} | {float(row['latency_gain_sum_vs_dense']):.2f} |"
        )
    if dense and best_speed:
        lines.extend(
            [
                "",
                "## Frontier Endpoints",
                "",
                f"- **Conservative (dense)**: quality={float(dense['quality_cost']):.6f}, "
                f"latency_ms={float(dense['latency_ms']):.2f}, "
                f"speedup={float(dense['speedup_vs_dense_linear']):.4f}, "
                f"methods={ {m: int(float(dense.get(f'count_{m}', 0))) for m in METHODS if int(float(dense.get(f'count_{m}', 0))) > 0} }",
                "",
                f"- **Speed**: quality={float(best_speed['quality_cost']):.6f}, "
                f"latency_ms={float(best_speed['latency_ms']):.2f}, "
                f"speedup={float(best_speed['speedup_vs_dense_linear']):.4f}, "
                f"methods={ {m: int(float(best_speed.get(f'count_{m}', 0))) for m in METHODS if int(float(best_speed.get(f'count_{m}', 0))) > 0} }",
                "",
                "## Frontier Progression",
                "",
            ]
        )
        for row in point_summary:
            counts = {m: int(float(row.get(f"count_{m}", 0))) for m in METHODS if int(float(row.get(f"count_{m}", 0))) > 0}
            lines.append(
                f"- Point {int(float(row['point_index']))}: quality={float(row['quality_cost']):.4f}, "
                f"latency={float(row['latency_ms']):.2f}ms, "
                f"speedup={float(row['speedup_vs_dense_linear']):.4f}, "
                f"prefill={float(row.get('total_prefill_ms', 0)):.1f}ms, "
                f"decode={float(row.get('total_decode_ms', 0)):.1f}ms, "
                f"conv={float(row.get('total_conversion_ms', 0)):.1f}ms, "
                f"counts={counts}"
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
