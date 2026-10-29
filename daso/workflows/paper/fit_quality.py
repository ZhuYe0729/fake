#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from daso.modeling.quality import FUSED_TYPES, QualityCalibrator, build_policy_features
from daso.policy import METHODS, PHASES, load_policy


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_errors(root: Path, buckets: int, scenario: str) -> dict[tuple[str, int, str, str], float]:
    output = {}
    for phase in PHASES:
        for method in METHODS:
            if method == "dense_bf16":
                for bucket in range(buckets):
                    for kind in FUSED_TYPES:
                        output[phase, bucket, kind, method] = 0.0
                continue
            path = root / f"{phase}_{method}.csv"
            if scenario == "prefill_only" and phase == "decode" and not path.exists():
                for bucket in range(buckets):
                    for kind in FUSED_TYPES:
                        output[phase, bucket, kind, method] = 0.0
                continue
            for row in read_csv(path):
                output[phase, int(row["layer_bucket"]), row["fused_type"], method] = \
                    float(row["output_rel_mse"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--local-errors", type=Path, required=True)
    parser.add_argument("--nll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--scenario", choices=("prefill_only", "prefill_decode"), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    labels = {row["policy_id"]: row for row in read_csv(args.nll)}
    buckets = args.layers // 8
    errors = load_errors(args.local_errors, buckets, args.scenario)
    policies = [load_policy(args.policy_dir / f"{row['policy_id']}.json", layers=args.layers)
                for row in manifest]
    features = torch.stack([build_policy_features(policy, errors, buckets=buckets)
                            for policy in policies])
    targets = torch.tensor([float(labels[row["policy_id"]]["target_delta_nll"])
                            for row in manifest], dtype=torch.float64)
    train = torch.tensor([row["split"] == "train" for row in manifest])
    fit = QualityCalibrator().fit(features, targets, train)
    args.output.mkdir(parents=True, exist_ok=True)
    fit.save(args.output / "model.json")
    predictions = fit.predict(features).tolist()
    rows = [{"policy_id": row["policy_id"], "split": row["split"],
             "actual_delta_nll": float(targets[index]),
             "predicted_delta_nll": predictions[index],
             "residual_actual_minus_predicted": float(targets[index]) - predictions[index]}
            for index, row in enumerate(manifest)]
    with (args.output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics = {"scenario": args.scenario}
    for split in ("train", "holdout"):
        subset = [row for row in rows if row["split"] == split]
        errors_abs = [abs(row["residual_actual_minus_predicted"]) for row in subset]
        squared = [row["residual_actual_minus_predicted"] ** 2 for row in subset]
        metrics[split] = {
            "count": len(subset),
            "mae": sum(errors_abs) / len(errors_abs),
            "rmse": math.sqrt(sum(squared) / len(squared)),
        }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
