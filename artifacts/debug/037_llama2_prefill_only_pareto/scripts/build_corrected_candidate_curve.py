#!/usr/bin/env python3
"""Attach the validated monotone E2E correction to the frozen 034 policies."""
from __future__ import annotations

import csv
import json
from pathlib import Path


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
    pairs = sorted(train)
    xs = [a for a, _ in pairs]
    ys = pava([b for _, b in pairs])
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            alpha = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + alpha * (ys[i] - ys[i - 1])
    raise AssertionError("unreachable")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    old = root.parent / "034_llama2_7b_chat_wikitext_pareto_solver"
    with (old / "prefill_only/pareto/pareto_points.csv").open(newline="") as handle:
        policies = list(csv.DictReader(handle))
    with (root / "e2e_calibration.csv").open(newline="") as handle:
        measured = list(csv.DictReader(handle))
    fit = [(float(r["raw_linear_ms"]), float(r["e2e_median_ms"])) for r in measured]
    known = {int(r["point_index"]): float(r["e2e_median_ms"]) for r in measured}
    rows = []
    for policy in policies:
        point = int(policy["point_index"])
        raw = float(policy["raw_predicted_linear_ms"])
        corrected = known.get(point, predict(fit, raw))
        row = dict(policy)
        row.update({
            "corrected_e2e_ms": corrected,
            "speed_axis": "measured" if point in known else "monotone_interpolated",
            "quality_proxy": "normalized_pooled_54train_18holdout_spearman_0.774",
        })
        rows.append(row)
    dense = next(float(r["corrected_e2e_ms"]) for r in rows if int(r["point_index"]) == 0)
    for row in rows:
        row["corrected_speedup_vs_dense"] = dense / float(row["corrected_e2e_ms"])
    output = root / "pareto/corrected_candidate_curve.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "policy_source": str(old / "prefill_only/pareto"),
        "quality_model": "normalized_pooled; 54 train / 18 holdout; holdout Spearman 0.774",
        "speed_model": "raw roofline-plus-local-residual kernel sum, then all-point monotone E2E correction",
        "optimization_invariance": "The correction is monotone non-decreasing in raw latency; for any fixed quality budget it preserves the latency-minimizing discrete policy. The frozen DP assignments are therefore retained and only the E2E axis changes.",
        "actual_speed_points": sorted(known),
        "nll_validation_points": [1, 3, 6, 9, 11, 13, 15],
    }
    (root / "pareto/corrected_candidate_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
