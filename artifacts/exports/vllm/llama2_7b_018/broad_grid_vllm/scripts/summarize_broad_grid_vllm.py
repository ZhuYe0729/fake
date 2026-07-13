#!/usr/bin/env python3
"""Summarize broad-grid vLLM benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BROAD_ROOT = SCRIPT_DIR.parent
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
INPUT_SEQS = (128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536)
OUTPUT_SEQS = (1, 16, 64, 128)
METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "hetero",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-long", type=Path, default=BROAD_ROOT / "results/summary_long.csv")
    parser.add_argument("--output-dir", type=Path, default=BROAD_ROOT / "summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.summary_long)
    by_key = {
        (row["method"], int(row["batch"]), int(row["input_seq"]), int(row["output_seq"])): row
        for row in rows
    }
    latency_rows = build_latency_table(by_key)
    speedup_rows = build_speedup_table(by_key)
    write_csv(args.output_dir / "broad_grid_latency_table.csv", latency_rows)
    write_csv(args.output_dir / "broad_grid_speedup_table.csv", speedup_rows)
    write_report(args.output_dir / "broad_grid_report.md", rows, latency_rows, speedup_rows)
    write_json(
        args.output_dir / "broad_grid_summary_metadata.json",
        {
            "config_count": len(BATCHES) * len(INPUT_SEQS) * len(OUTPUT_SEQS),
            "methods": list(METHODS),
            "summary_long": str(args.summary_long),
        },
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_latency_table(by_key: dict[tuple[str, int, int, int], dict[str, str]]) -> list[dict[str, Any]]:
    table = []
    for batch, input_seq, output_seq in all_configs():
        row: dict[str, Any] = {
            "batch": batch,
            "input_seq": input_seq,
            "output_seq": output_seq,
        }
        for method in METHODS:
            item = by_key.get((method, batch, input_seq, output_seq))
            row[method] = latency_cell(item)
        table.append(row)
    return table


def build_speedup_table(by_key: dict[tuple[str, int, int, int], dict[str, str]]) -> list[dict[str, Any]]:
    table = []
    for batch, input_seq, output_seq in all_configs():
        row: dict[str, Any] = {
            "batch": batch,
            "input_seq": input_seq,
            "output_seq": output_seq,
        }
        dense = by_key.get(("dense_bf16", batch, input_seq, output_seq))
        dense_ms = median_ms(dense)
        for method in METHODS:
            if method == "dense_bf16":
                row[method] = "1.000" if dense_ms is not None else latency_cell(dense)
                continue
            item = by_key.get((method, batch, input_seq, output_seq))
            item_ms = median_ms(item)
            if dense_ms is None or item_ms is None or item_ms == 0:
                row[method] = latency_cell(item)
            else:
                row[method] = f"{dense_ms / item_ms:.3f}"
        table.append(row)
    return table


def all_configs():
    for batch in BATCHES:
        for input_seq in INPUT_SEQS:
            for output_seq in OUTPUT_SEQS:
                yield batch, input_seq, output_seq


def latency_cell(row: dict[str, str] | None) -> str:
    if row is None:
        return "MISSING"
    if row.get("status") != "OK":
        return row.get("status") or "ERROR"
    value = median_ms(row)
    return "ERROR" if value is None else f"{value:.3f}"


def median_ms(row: dict[str, str] | None) -> float | None:
    if row is None or row.get("status") != "OK":
        return None
    try:
        value = float(row["median_ms"])
    except (KeyError, TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def write_report(
    path: Path,
    rows: list[dict[str, str]],
    latency_rows: list[dict[str, Any]],
    speedup_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Llama2-7B Broad Grid vLLM Benchmark",
        "",
        "Latency cells are median wall-clock latency in milliseconds. Failed cells keep their status label.",
        "Speedup cells are relative to `dense_bf16` for the same `(batch,input_seq,output_seq)`.",
        "",
        "## Grid",
        "",
        f"- Batch values: {', '.join(map(str, BATCHES))}",
        f"- Input seq values: {', '.join(map(str, INPUT_SEQS))}",
        f"- Output seq values: {', '.join(map(str, OUTPUT_SEQS))}",
        f"- Config rows: {len(latency_rows)}",
        f"- Raw method-config rows: {len(rows)}",
        "",
        "## Method Status Counts",
        "",
        "| method | OK | PRECHECK_OOM | INIT_OOM | OOM | OOM_SKIPPED | ERROR/MISSING |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        method_rows = [row for row in rows if row.get("method") == method]
        counts = Counter(row.get("status", "MISSING") for row in method_rows)
        missing = len(BATCHES) * len(INPUT_SEQS) * len(OUTPUT_SEQS) - len(method_rows)
        if missing > 0:
            counts["MISSING"] += missing
        error_missing = sum(
            count
            for status, count in counts.items()
            if status not in {"OK", "PRECHECK_OOM", "INIT_OOM", "OOM", "OOM_SKIPPED"}
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(counts.get("OK", 0)),
                    str(counts.get("PRECHECK_OOM", 0)),
                    str(counts.get("INIT_OOM", 0)),
                    str(counts.get("OOM", 0)),
                    str(counts.get("OOM_SKIPPED", 0)),
                    str(error_missing),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Average Speedup Over Comparable OK Cells", ""])
    lines.append("| method | comparable_configs | avg_speedup |")
    lines.append("|---|---:|---:|")
    for method in METHODS:
        values = []
        for row in speedup_rows:
            if method == "dense_bf16":
                continue
            if row.get("dense_bf16") != "1.000":
                continue
            try:
                values.append(float(row[method]))
            except (KeyError, TypeError, ValueError):
                pass
        avg = sum(values) / len(values) if values else math.nan
        lines.append(f"| {method} | {len(values)} | {avg:.3f} |" if values else f"| {method} | 0 | pending |")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `results/summary_long.csv`: long-form method-config summary.",
            "- `results/iterations.csv`: raw timed iterations.",
            "- `summary/broad_grid_latency_table.csv`: requested wide latency table.",
            "- `summary/broad_grid_speedup_table.csv`: wide speedup table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
