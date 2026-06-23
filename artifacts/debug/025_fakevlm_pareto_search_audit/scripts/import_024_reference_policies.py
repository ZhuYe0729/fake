#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from common_search_audit import DEBUG_ROOT, DEFAULT_BATCH_SIZE, SOURCE_024_ROOT, f, read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register selected 024 Pareto policies for 025 validation.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--reference-report", type=Path, default=None, help="Optional 024 report CSV whose pareto rows replace selected_pareto_points.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_path = args.output_root / "search" / "search_policies.csv"
    rows = read_csv(search_path)
    rows = [row for row in rows if row.get("family") != "reference_024"]
    if args.reference_report is None:
        reference_rows = make_reference_rows(args.output_root, args.batch_size)
    else:
        reference_rows = make_reference_rows_from_report(args.output_root, args.batch_size, args.reference_report)
    write_csv(search_path, rows + reference_rows)
    write_csv(args.output_root / "search" / "reference_024_policies.csv", reference_rows)
    print(f"registered reference_024 rows={len(reference_rows)}")


def make_reference_rows(output_root: Path, batch_size: int) -> list[dict[str, Any]]:
    selected_path = SOURCE_024_ROOT / "validation" / "selected_pareto_points.csv"
    selected = [row for row in read_csv(selected_path) if int(f(row, "batch_size")) == batch_size]
    out = []
    for row in selected:
        point_index = int(f(row, "point_index"))
        key = f"reference_024_batch_{batch_size}_point_{point_index:03d}"
        source_policy = SOURCE_024_ROOT.parents[2] / row["policy_json"]
        dest_policy = output_root / "policies" / "reference_024" / f"batch_{batch_size}_point_{point_index:03d}.json"
        dest_policy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_policy, dest_policy)
        counts = {
            name: int(f(row, f"count_{name}"))
            for name in ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
        }
        dense_latency = f(row, "dense_latency_ms")
        latency = f(row, "latency_ms")
        out.append(
            {
                "key": key,
                "family": "reference_024",
                "policy_name": f"batch_{batch_size}_point_{point_index:03d}",
                "policy_json": str(dest_policy.resolve()),
                "batch_size": batch_size,
                "predicted_linear_latency_ms": latency,
                "predicted_linear_speedup_vs_dense": dense_latency / latency if latency > 0 else "",
                "predicted_quality_cost": f(row, "quality_cost"),
                "backend_counts": str(counts),
                "replacement_ratio": "",
                "count_dense_bf16": counts["dense_bf16"],
                "count_dense_nvfp4": counts["dense_nvfp4"],
                "count_sparse_bf16": counts["sparse_bf16"],
                "count_sparse_nvfp4": counts["sparse_nvfp4"],
                "parent_point": point_index,
                "mutation_rate": "",
                "suspicious_module_count": "",
            }
        )
    return out


def make_reference_rows_from_report(output_root: Path, batch_size: int, report_path: Path) -> list[dict[str, Any]]:
    report_rows = [
        row
        for row in read_csv(report_path)
        if row.get("row_type") == "pareto" and int(f(row, "batch_size")) == batch_size
    ]
    out = []
    for row in report_rows:
        point_index = int(f(row, "point_index"))
        key = f"reference_024_batch_{batch_size}_point_{point_index:03d}"
        source_policy = SOURCE_024_ROOT.parents[2] / row["policy_json"]
        dest_policy = output_root / "policies" / "reference_024" / f"batch_{batch_size}_point_{point_index:03d}.json"
        dest_policy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_policy, dest_policy)
        counts = {
            name: int(f(row, f"count_{name}"))
            for name in ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
        }
        out.append(
            {
                "key": key,
                "family": "reference_024",
                "policy_name": f"batch_{batch_size}_point_{point_index:03d}",
                "policy_json": str(dest_policy.resolve()),
                "batch_size": batch_size,
                "predicted_linear_latency_ms": f(row, "predicted_linear_latency_ms"),
                "predicted_linear_speedup_vs_dense": f(row, "predicted_linear_speedup"),
                "predicted_quality_cost": f(row, "predicted_quality_cost"),
                "backend_counts": str(counts),
                "replacement_ratio": "",
                "count_dense_bf16": counts["dense_bf16"],
                "count_dense_nvfp4": counts["dense_nvfp4"],
                "count_sparse_bf16": counts["sparse_bf16"],
                "count_sparse_nvfp4": counts["sparse_nvfp4"],
                "parent_point": point_index,
                "mutation_rate": "",
                "suspicious_module_count": "",
            }
        )
    return out


if __name__ == "__main__":
    main()
