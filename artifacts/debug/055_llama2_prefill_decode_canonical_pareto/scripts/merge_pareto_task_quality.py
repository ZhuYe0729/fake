#!/usr/bin/env python3
"""Merge canonical Pareto task shards and compute PMPD metrics."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat"
KIT = ROOT / "references/pmpd_eval_kit/pmpd_eval.py"
PYTHON = "/home/agent/wja/miniconda3/envs/vllm/bin/python"
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default="point_001,point_002,point_003,point_004,point_005,point_006,point_008,point_009")
    args = parser.parse_args()
    policies = tuple(item.strip() for item in args.policies.split(",") if item.strip())
    task = EXP / "task_quality"
    predicted = {row["policy_id"]: row for row in csv.DictReader((EXP / "pareto/predicted_points.csv").open())}
    closure = {row["policy_id"]: row for row in csv.DictReader((EXP / "validation/closure_summary.csv").open())}
    rows = []
    for policy in policies:
        label = f"ours_{policy}_prefill_decode"
        for dataset, expected in DATASETS.items():
            records: dict[str, dict] = {}
            for path in sorted((task / "shards" / policy / dataset).glob("shard_*/**/*.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line); key = str(item["question_id"])
                    if key in records: raise RuntimeError(f"duplicate {policy}/{dataset}/{key}")
                    records[key] = item
            if len(records) != expected: raise RuntimeError(f"incomplete {policy}/{dataset}: {len(records)} != {expected}")
            target = task / "results" / policy / dataset / f"{label}-fp16.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            ordered = sorted(records.values(), key=lambda item: int(item["question_id"]))
            target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")
            metrics_path = target.with_name("metrics.json")
            if not metrics_path.exists():
                subprocess.run([PYTHON, str(KIT), "--dataset", dataset, "--metrics-only", str(target)], check=True)
            metrics = json.loads(metrics_path.read_text())
            model = predicted[policy]; actual = closure.get(policy, {})
            rows.append({"policy_id": policy, "dataset": dataset, "samples": metrics["num_samples"],
                         "predicted_delta_nll": model["predicted_delta_nll"],
                         "measured_delta_nll": actual.get("measured_delta_nll", ""),
                         "measured_speedup_vs_dense": actual.get("measured_speedup_vs_dense", ""),
                         "raw_predicted_speedup_vs_dense": model["raw_speedup_vs_dense"],
                         "rougeL_percent": metrics.get("rougeL_percent", ""),
                         "bert_score_percent": metrics.get("bert_score_percent", ""),
                         "sacre_bleu": metrics.get("sacre_bleu", ""),
                         "empty_predictions": metrics["empty_predictions"]})
    out = task / "summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
