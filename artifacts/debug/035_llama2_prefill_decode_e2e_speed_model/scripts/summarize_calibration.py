#!/usr/bin/env python3
"""Summarize formal E2E calibration measurements without fitting a new solver yet."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root.parent / "034_llama2_7b_chat_wikitext_pareto_solver"
    predicted = list(csv.DictReader((source / "prefill_decode/pareto/pareto_points.csv").open()))
    predicted_by_point = {int(row["point_index"]): row for row in predicted}
    rows = []
    for directory in sorted((root / "measurements").glob("point_*")):
        point = int(directory.name.split("_")[1])
        runs = directory / "runs"
        full = sorted(runs.glob("measured_*_o80.json"))
        one = sorted(runs.glob("measured_*_o1.json"))
        if not full or len(one) != len(full):
            continue
        e2e = statistics.median(json.loads(path.read_text())["elapsed_ms"] for path in full)
        ttft = statistics.median(json.loads(path.read_text())["elapsed_ms"] for path in one)
        row = predicted_by_point[point]
        rows.append({
            "point_index": point, "samples": len(full), "formal_e2e_ms": e2e,
            "formal_ttft_ms": ttft, "formal_tpot_ms": (e2e - ttft) / 79,
            "raw_predicted_linear_ms": row["raw_predicted_linear_ms"],
            "raw_linear_speedup_vs_dense": row["raw_linear_speedup_vs_dense"],
        })
    out = root / "calibration_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["point_index"])
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
