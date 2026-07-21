#!/usr/bin/env python3
"""Fit a train-only monotone E2E correction over reused roofline predictions."""
from __future__ import annotations
import csv
import json
import statistics
from scenario import EXP

def pava(values: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop(); blocks[-1][0] += right[0]; blocks[-1][1] += right[1]
    return [total / count for total, count in blocks for _ in range(int(count))]

def interpolate(train: list[tuple[float, float]], x: float) -> float:
    pairs = sorted(train); xs = [a for a, _ in pairs]; ys = pava([b for _, b in pairs])
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for index in range(1, len(xs)):
        if x <= xs[index]:
            ratio = (x - xs[index-1]) / (xs[index] - xs[index-1])
            return ys[index-1] + ratio * (ys[index] - ys[index-1])
    raise AssertionError

def main() -> None:
    root = EXP / "speed/calibration"
    design = list(csv.DictReader((root / "design.csv").open()))
    metadata = json.loads((root / "metadata.json").read_text()); train_ids = set(metadata["train_policies"])
    rows = []
    for row in design:
        values = [json.loads((root / "runs" / row["policy_id"] / f"measured_{i}.json").read_text())["elapsed_ms"] for i in range(5)]
        rows.append({**row, "e2e_median_ms": statistics.median(values), "raw_runs_ms": json.dumps(values),
                     "split": "train" if row["policy_id"] in train_ids else "holdout"})
    dense = next(row for row in rows if row["policy_id"] == "p00")
    scale = float(dense["e2e_median_ms"]) / float(dense["raw_predicted_linear_ms"])
    train = [(float(row["raw_predicted_linear_ms"]), float(row["e2e_median_ms"])) for row in rows if row["split"] == "train"]
    for row in rows:
        raw = float(row["raw_predicted_linear_ms"])
        row["dense_scaled_prediction_ms"] = raw * scale
        row["monotone_prediction_ms"] = interpolate(train, raw)
    holdout = [row for row in rows if row["split"] == "holdout"]
    mae = lambda key: sum(abs(float(r["e2e_median_ms"]) - float(r[key])) for r in holdout) / len(holdout)
    metrics = {"holdout": [r["policy_id"] for r in holdout], "raw_dense_scaled_mae_ms": mae("dense_scaled_prediction_ms"),
               "monotone_mae_ms": mae("monotone_prediction_ms"), "monotone_improves": mae("monotone_prediction_ms") < mae("dense_scaled_prediction_ms")}
    with (root / "calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__": main()
