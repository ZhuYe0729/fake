#!/usr/bin/env python3
"""Fit the paper's monotone E2E correction over calibrated Roofline totals."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from daso.modeling.latency import monotonic_calibration
from daso.policy import load_policy


KERNEL = {"dense_bf16": "dense_bf16", "dense_nvfp4": "dense_nvfp4",
          "sparse_bf16": "sparse_bf16", "sparse_nvfp4": "sparse_nvfp4",
          "w4a16_ours": "marlin_nvfp4"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def raw_latency(policy: dict, actions: dict[tuple[str, str, str], float],
                scenario: str, decode_steps: int) -> float:
    total = 0.0
    for name, methods in policy["method_map"].items():
        total += actions["prefill", name, KERNEL[methods["prefill_method"]]]
        if scenario == "prefill_decode":
            total += decode_steps * actions["decode", name, KERNEL[methods["decode_method"]]]
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True,
                        help="CSV: policy_id,policy_json,measured_ms (or e2e_median_ms),split")
    parser.add_argument("--policy-dir", type=Path,
                        help="relocate policy_json basenames here (useful for archived designs)")
    parser.add_argument("--latency-actions", type=Path, required=True)
    parser.add_argument("--scenario", choices=("prefill_only", "prefill_decode"), required=True)
    parser.add_argument("--decode-steps", type=int, default=63)
    parser.add_argument("--dense-policy", default="p00")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actions = {(row["phase"], row["module_name"], row["kernel"]): float(row["latency_ms"])
               for row in read(args.latency_actions)
               if row["supported"].lower() == "true" and row["latency_ms"]}
    rows = read(args.design)
    for row in rows:
        policy_path = ((args.policy_dir / Path(row["policy_json"]).name)
                       if args.policy_dir is not None else Path(row["policy_json"]))
        policy = load_policy(policy_path)
        row["raw_predicted_ms"] = raw_latency(policy, actions, args.scenario, args.decode_steps)
        measured = row.get("measured_ms") or row.get("e2e_median_ms")
        if measured is None:
            raise ValueError("design requires measured_ms or e2e_median_ms")
        row["measured_ms"] = float(measured)
    dense = next(row for row in rows if row["policy_id"] == args.dense_policy)
    scale = dense["measured_ms"] / dense["raw_predicted_ms"]

    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            grouped[row["raw_predicted_ms"]].append(row["measured_ms"])
    xs = sorted(grouped)
    raw_ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    ys = monotonic_calibration(xs, raw_ys)

    def interpolate(x: float) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for index in range(1, len(xs)):
            if x <= xs[index]:
                ratio = (x - xs[index - 1]) / (xs[index] - xs[index - 1])
                return ys[index - 1] + ratio * (ys[index] - ys[index - 1])
        raise AssertionError

    for row in rows:
        row["dense_scaled_prediction_ms"] = row["raw_predicted_ms"] * scale
        row["monotone_prediction_ms"] = interpolate(row["raw_predicted_ms"])
    metrics = {}
    for split in ("train", "holdout"):
        subset = [row for row in rows if row["split"] == split]
        metrics[split] = {}
        for key in ("dense_scaled_prediction_ms", "monotone_prediction_ms"):
            absolute = [abs(row[key] - row["measured_ms"]) for row in subset]
            relative = [error / row["measured_ms"] for error, row in zip(absolute, subset)]
            metrics[split][key] = {
                "mae_ms": sum(absolute) / len(absolute),
                "rmse_ms": math.sqrt(sum(error * error for error in absolute) / len(absolute)),
                "mape_percent": 100 * sum(relative) / len(relative),
            }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output / "model.json").write_text(json.dumps({
        "scenario": args.scenario, "decode_steps": args.decode_steps,
        "dense_scale": scale, "monotone_x_ms": xs, "monotone_y_ms": ys,
    }, indent=2) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
