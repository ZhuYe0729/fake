#!/usr/bin/env python3
"""Summarize real canonical NLL and fresh-process speed closure points."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat"
BASE_NLL = 1.920185718948398  # canonical p00, all dense BF16


def without_slow_outliers(samples: list[float]) -> list[float]:
    """Keep the raw JSON intact but exclude only clearly interfered slow runs."""
    median = statistics.median(samples)
    return [sample for sample in samples if sample <= 1.15 * median]


def main() -> None:
    predicted = {row["policy_id"]: row for row in csv.DictReader((EXP / "pareto/predicted_points.csv").open())}
    rows = []
    for policy_id, row in predicted.items():
        path = EXP / "validation/nll" / f"{policy_id}.json"
        if not path.exists():
            continue
        nll = json.loads(path.read_text())
        rows.append({"policy_id": policy_id, "predicted_delta_nll": float(row["predicted_delta_nll"]),
                     "measured_delta_nll": float(nll["avg_nll"]) - BASE_NLL,
                     "nll_prediction_error": float(nll["avg_nll"]) - BASE_NLL - float(row["predicted_delta_nll"])})
    by_id = {row["policy_id"]: row for row in rows}
    speed_groups = {
        "runs_util080": ("point_001", "point_002", "point_003", "point_005", "point_006", "point_008", "point_009"),
        "runs_util080_gpu7clean": ("point_004",),
    }
    for group, policy_ids in speed_groups.items():
        baseline_samples = [json.loads(path.read_text())["elapsed_ms"]
                            for path in sorted((EXP / "validation/speed/point_000" / group).glob("measured_*_o80.json"))]
        baseline = statistics.median(without_slow_outliers(baseline_samples))
        for policy_id in policy_ids:
            runs = EXP / "validation/speed" / policy_id / group
            full = [json.loads(path.read_text())["elapsed_ms"] for path in sorted(runs.glob("measured_*_o80.json"))]
            ttft = [json.loads(path.read_text())["elapsed_ms"] for path in sorted(runs.glob("measured_*_o1.json"))]
            if not full or not ttft:
                continue
            valid_full = without_slow_outliers(full)
            valid_ttft = without_slow_outliers(ttft)
            row = by_id.get(policy_id)
            if row is None:
                row = {"policy_id": policy_id, "predicted_delta_nll": float(predicted[policy_id]["predicted_delta_nll"]),
                       "measured_delta_nll": "", "nll_prediction_error": ""}
                rows.append(row); by_id[policy_id] = row
            median = statistics.median(valid_full)
            row.update({"speed_config": "util080",
                        "speed_anchor_group": group,
                        "raw_speed_samples": len(full), "speed_samples": len(valid_full),
                        "formal_e2e_median_ms": median,
                        "formal_ttft_median_ms": statistics.median(valid_ttft),
                        "formal_tpot_ms": (median - statistics.median(valid_ttft)) / 79,
                        "formal_e2e_cv": statistics.stdev(valid_full) / statistics.mean(valid_full) if len(valid_full) > 1 else 0.0,
                        "measured_speedup_vs_dense": baseline / median,
                        "raw_predicted_speedup_vs_dense": float(predicted[policy_id]["raw_speedup_vs_dense"]),
                        "calibrated_predicted_speedup_vs_dense": float(predicted[policy_id]["calibrated_speedup_vs_dense"])})
    out = EXP / "validation/closure_summary.csv"
    fields = sorted({key for row in rows for key in row})
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
