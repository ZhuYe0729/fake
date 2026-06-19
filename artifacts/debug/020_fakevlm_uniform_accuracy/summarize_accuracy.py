#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = (
    "dense_bf16",
    "sparse_bf16",
    "dense_nvfp4",
    "sparse_nvfp4",
    "marlin_weight_only",
    "dense_nvfp4_prefill_marlin_decode",
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FakeVLM uniform accuracy outputs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for method in METHODS:
        accuracy_path = args.output_root / "outputs" / method / "accuracy.json"
        prepare_path = args.output_root / "compression" / method / "prepare_metadata.json"
        status_path = args.output_root / "status" / f"{method}.json"
        row = {"method": method}
        if accuracy_path.exists():
            accuracy = read_json(accuracy_path)
            row.update(
                {
                    "status": "ok",
                    "total_right": accuracy["global_stats"]["total_right"],
                    "total_wrong": accuracy["global_stats"]["total_wrong"],
                    "global_accuracy": f"{accuracy['global_stats']['global_accuracy']:.6f}",
                }
            )
        else:
            status = read_json(status_path) if status_path.exists() else {}
            row.update(
                {
                    "status": "failed",
                    "total_right": "",
                    "total_wrong": "",
                    "global_accuracy": "",
                    "error": status.get("error", ""),
                    "exit_code": status.get("exit_code", ""),
                }
            )
        if prepare_path.exists():
            prepare = read_json(prepare_path)
            row.update(
                {
                    "replacement_backend": prepare.get("replacement_backend", ""),
                    "replaced_linear_count": prepare.get("replaced_linear_count", ""),
                    "skipped_linear_count": prepare.get("skipped_linear_count", ""),
                    "activation_quant": prepare.get("activation_quant", ""),
                    "calibration_used": prepare.get("calibration_used", ""),
                    "prepare_elapsed_sec": prepare.get("elapsed_sec", ""),
                }
            )
        rows.append(row)

    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "accuracy_summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(summary_dir / "accuracy_summary.md", rows)
    print(f"wrote {csv_path}")


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "method",
        "status",
        "global_accuracy",
        "total_right",
        "total_wrong",
        "replacement_backend",
        "replaced_linear_count",
        "activation_quant",
        "calibration_used",
    ]
    lines = ["# FakeVLM Uniform Accuracy Summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
