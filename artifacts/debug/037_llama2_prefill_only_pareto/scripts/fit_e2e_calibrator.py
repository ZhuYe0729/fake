#!/usr/bin/env python3
"""Fit and validate a monotone policy-level E2E correction for prefill-only."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


NEW_POINTS = {1, 3, 6, 9, 11, 13, 15}
OLD_POINTS = {0, 4, 8, 12, 16}


def pava(values: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            blocks[-1][0] += right[0]
            blocks[-1][1] += right[1]
    return [total / count for total, count in blocks for _ in range(int(count))]


def predict(train: list[tuple[float, float]], x: float) -> float:
    ordered = sorted(train)
    xs = [a for a, _ in ordered]
    ys = pava([b for _, b in ordered])
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    raise AssertionError("unreachable")


def median(runs: Path) -> float:
    values = [json.loads(path.read_text())["elapsed_ms"] for path in sorted(runs.glob("measured_*.json"))]
    if len(values) != 5:
        raise ValueError(f"expected five runs in {runs}, found {len(values)}")
    return statistics.median(values)


def mae(rows: list[dict[str, float]], key: str) -> float:
    return statistics.mean(abs(row["e2e_median_ms"] - row[key]) for row in rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    old = root.parent / "034_llama2_7b_chat_wikitext_pareto_solver"
    policies = list(csv.DictReader((old / "prefill_only/pareto/pareto_points.csv").open()))
    raw = {int(row["point_index"]): float(row["raw_predicted_linear_ms"]) for row in policies}
    rows: list[dict[str, float | int | str]] = []
    for point in sorted(NEW_POINTS | OLD_POINTS):
        runs = root / "measurements" / f"point_{point}" / "runs"
        source = "new_calibration" if point in NEW_POINTS else "034_heldout"
        if point in OLD_POINTS:
            runs = old / "validation/prefill_only/speed" / f"point_{point}" / "runs"
        rows.append({"point_index": point, "source": source, "raw_linear_ms": raw[point],
                     "e2e_median_ms": median(runs)})

    train_new = [(float(r["raw_linear_ms"]), float(r["e2e_median_ms"])) for r in rows if r["point_index"] in NEW_POINTS]
    all_pairs = [(float(r["raw_linear_ms"]), float(r["e2e_median_ms"])) for r in rows]
    dense = next(r for r in rows if r["point_index"] == 0)
    scale = float(dense["e2e_median_ms"]) / float(dense["raw_linear_ms"])
    for row in rows:
        x = float(row["raw_linear_ms"])
        row["raw_dense_scaled_ms"] = x * scale
        row["strict_holdout_monotone_ms"] = predict(train_new, x)
        loo = [(a, b) for a, b in all_pairs if a != x]
        row["all_point_loo_monotone_ms"] = predict(loo, x)

    heldout = [r for r in rows if r["point_index"] in OLD_POINTS]
    metrics = {
        "points": len(rows),
        "new_calibration_points": sorted(NEW_POINTS),
        "strict_heldout_points": sorted(OLD_POINTS),
        "strict_heldout_raw_dense_scaled_mae_ms": mae(heldout, "raw_dense_scaled_ms"),
        "strict_heldout_monotone_mae_ms": mae(heldout, "strict_holdout_monotone_ms"),
        "all_point_loo_raw_dense_scaled_mae_ms": mae(rows, "raw_dense_scaled_ms"),
        "all_point_loo_monotone_mae_ms": mae(rows, "all_point_loo_monotone_ms"),
        "status": "speed-model-validation-only; frontier re-solving remains TODO",
    }
    with (root / "e2e_calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (root / "e2e_calibration_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
