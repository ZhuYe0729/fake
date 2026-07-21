#!/usr/bin/env python3
"""Merge all real-vLLM generation shards and compute task metrics."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = Path(os.environ["COSPAQ_EXPERIMENT_DIR"])
KIT = ROOT / "references/pmpd_eval_kit/pmpd_eval.py"
PYTHON = "/home/agent/wja/miniconda3/envs/vllm/bin/python"
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
DEFAULT = "p00,p01,p02,p03,p04," + ",".join(f"point_{i:03d}" for i in range(12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default=DEFAULT)
    args = parser.parse_args()
    policies = [item for item in args.policies.split(",") if item]
    predicted = {row["policy_id"]: row for row in csv.DictReader((EXP / "pareto/predicted_points.csv").open())}
    closure = {row["policy_id"]: row for row in csv.DictReader((EXP / "pareto/closure_summary.csv").open())}
    uniform = {row["policy_id"]: row for row in csv.DictReader((EXP / "speed/uniform_baselines.csv").open())}
    task_root = EXP / "task_quality"; rows = []
    for policy_id in policies:
        label = f"phase_{policy_id}_prefill_decode"
        for dataset, expected in DATASETS.items():
            records: dict[str, dict] = {}
            for path in sorted((task_root / "shards" / policy_id / dataset).glob("shard_*/**/*.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line); key = str(item["question_id"])
                    if key in records: raise RuntimeError(f"duplicate {policy_id}/{dataset}/{key}")
                    records[key] = item
            if len(records) != expected:
                raise RuntimeError(f"incomplete {policy_id}/{dataset}: {len(records)} != {expected}")
            result = task_root / "results" / policy_id / dataset / f"{label}-fp16.jsonl"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in
                                      sorted(records.values(), key=lambda item: int(item["question_id"]))), encoding="utf-8")
            metrics_file = result.with_name("metrics.json")
            subprocess.run([PYTHON, str(KIT), "--dataset", dataset, "--metrics-only", str(result)], check=True)
            metrics = json.loads(metrics_file.read_text())
            if policy_id in predicted:
                speed = closure[policy_id]; kind = "ours"; pred_nll = predicted[policy_id]["predicted_delta_nll"]
                speedup = speed["measured_speedup_vs_dense"]
            else:
                speed = uniform[policy_id]; kind = "uniform"; pred_nll = ""; speedup = speed["speedup_vs_dense_bf16"]
            rows.append({"policy_id": policy_id, "kind": kind, "dataset": dataset,
                         "samples": metrics["num_samples"], "predicted_delta_nll": pred_nll,
                         "measured_speedup_vs_dense": speedup,
                         "rougeL_percent": metrics.get("rougeL_percent", ""),
                         "bert_score_percent": metrics.get("bert_score_percent", ""),
                         "sacre_bleu": metrics.get("sacre_bleu", ""),
                         "empty_predictions": metrics["empty_predictions"]})
    out = task_root / "summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
