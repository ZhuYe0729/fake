#!/usr/bin/env python3
"""Fit and score monotone E2E calibration over roofline kernel-cost sums."""
from __future__ import annotations
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat/speed/calibration"


def pava(values: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop(); blocks[-1][0] += right[0]; blocks[-1][1] += right[1]
    return [total / count for total, count in blocks for total, count in blocks for _ in range(int(count))]


def predict(train: list[tuple[float, float]], x: float) -> float:
    pairs = sorted(train); xs, ys = [a for a, _ in pairs], pava([b for _, b in pairs])
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for index in range(1, len(xs)):
        if x <= xs[index]:
            return ys[index - 1] + (x - xs[index - 1]) / (xs[index] - xs[index - 1]) * (ys[index] - ys[index - 1])
    raise AssertionError("unreachable")


def main() -> None:
    design = list(csv.DictReader((EXP / "design.csv").open()))
    rows = []
    missing = []
    for item in design:
        values = [json.loads(path.read_text())["elapsed_ms"] for path in sorted((EXP / "runs" / item["policy_id"]).glob("measured_*.json"))]
        if not values:
            missing.append(item["policy_id"]); continue
        if len(values) != 5: raise RuntimeError(f"{item['policy_id']} has {len(values)} measured samples")
        rows.append({**item, "e2e_median_ms": statistics.median(values), "e2e_samples_ms": json.dumps(values)})
    train_ids = {item["policy_id"] for item in design[:7]}
    if not train_ids <= {row["policy_id"] for row in rows}:
        raise RuntimeError("a required training calibration point is missing")
    train = [(float(row["raw_predicted_linear_ms"]), float(row["e2e_median_ms"])) for row in rows if row["policy_id"] in train_ids]
    dense = next(row for row in rows if row["policy_id"] == "p00")
    scale = float(dense["e2e_median_ms"]) / float(dense["raw_predicted_linear_ms"])
    for row in rows:
        raw = float(row["raw_predicted_linear_ms"])
        row["split"] = "train" if row["policy_id"] in train_ids else "holdout"
        row["raw_dense_scaled_ms"] = raw * scale
        row["monotone_prediction_ms"] = predict(train, raw)
    holdout = [row for row in rows if row["split"] == "holdout"]
    mae = lambda key: sum(abs(float(row["e2e_median_ms"]) - float(row[key])) for row in holdout) / len(holdout)
    with (EXP / "calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (EXP / "metrics.json").write_text(json.dumps({"holdout": [row["policy_id"] for row in holdout], "missing_planned_holdout": missing,
        "raw_dense_scaled_holdout_mae_ms": mae("raw_dense_scaled_ms"), "monotone_holdout_mae_ms": mae("monotone_prediction_ms"),
        "monotone_improves_holdout": mae("monotone_prediction_ms") < mae("raw_dense_scaled_ms")}, indent=2) + "\n")


if __name__ == "__main__": main()
