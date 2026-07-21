#!/usr/bin/env python3
"""Summarize common-runtime speed closure for solved Pareto policies."""
from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "llama2_7b_chat"))


def row(policy_id: str) -> dict[str, str]:
    path = EXP / "speed/runs" / policy_id / "summary.csv"
    return next(csv.DictReader(path.open()))


def main() -> None:
    predicted_path = EXP / "pareto/candidates/predicted_points.csv"
    if not predicted_path.exists():
        predicted_path = EXP / "pareto/predicted_points.csv"
    predicted = list(csv.DictReader(predicted_path.open()))
    measured = {item["policy_id"]: row(item["policy_id"]) for item in predicted}
    # Older Llama-2 solver outputs name the all-BF16 anchor ``b8o64000``.
    # Newer direct-layout outputs use ``point_000``.  Resolve the old name
    # first, then fall back to the unique predicted 1x-speed anchor.
    dense_policy = "b8o64000" if "b8o64000" in measured else next(
        item["policy_id"] for item in predicted
        if abs(float(item["raw_speedup_vs_dense"]) - 1.0) < 1e-9
    )
    dense = float(measured[dense_policy]["e2e_median_ms"])
    out = []
    for item in predicted:
        policy_id = item["policy_id"]; speed = measured[policy_id]
        out.append({"policy_id": policy_id, "e2e_median_ms": speed["e2e_median_ms"],
                    "ttft_median_ms": speed["ttft_median_ms"],
                    "measured_speedup_vs_dense": dense / float(speed["e2e_median_ms"]),
                    "raw_predicted_speedup_vs_dense": item["raw_speedup_vs_dense"],
                    "predicted_delta_nll": item["predicted_delta_nll"]})
    path = EXP / "pareto/closure_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0])); writer.writeheader(); writer.writerows(out)
    print(path)


if __name__ == "__main__":
    main()
