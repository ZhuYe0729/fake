#!/usr/bin/env python3
"""Summarize batch wall times already recorded by the continuous task runner."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POINTS = (34, 36, 37, 38)
DATASETS = ("cnn_dm_1000", "dsum", "IWSLT")


def main() -> None:
    base = ROOT / "task_quality_intermediate/shards"
    rows = []
    for point in POINTS:
        label = f"ours_intermediate_{point}_prefill_decode-fp16.jsonl"
        for dataset in DATASETS:
            values = []
            for path in (base / f"point_{point}" / dataset).glob(f"shard_*/{dataset}/{label}"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    values.append(float(item["choices"][0]["wall_time"][0]) * 1000.0)
            rows.append({"point": point, "dataset": dataset, "requests": len(values),
                         "batch_wall_median_ms": statistics.median(values), "batch_wall_mean_ms": statistics.mean(values),
                         "batch_wall_p90_ms": statistics.quantiles(values, n=10)[8]})
    out = ROOT / "task_quality_intermediate/continuous_runtime_summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
