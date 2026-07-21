#!/usr/bin/env python3
"""Merge the completed uniform p03/p04 shards and compute PMPD metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat"
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}


def merge(policy: str, dataset: str, expected: int) -> dict[str, object]:
    label = f"uniform_{policy}_prefill_decode"
    records: dict[str, dict] = {}
    shard_root = EXP / "task_quality/shards" / policy / dataset
    for path in sorted(shard_root.glob("shard_*/**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            key = str(item["question_id"])
            if key in records:
                raise RuntimeError(f"duplicate {policy}/{dataset}/{key}")
            records[key] = item
    if len(records) != expected:
        raise RuntimeError(f"incomplete {policy}/{dataset}: {len(records)} != {expected}")
    target = EXP / "task_quality/results" / policy / dataset / f"{label}-fp16.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(records[key], ensure_ascii=False) + "\n"
                              for key in sorted(records, key=int)), encoding="utf-8")
    raw_predictions = [item["choices"][0]["turns"][0] for item in records.values()]
    predictions = [item if item.strip() else "." for item in raw_predictions]
    references = [item["reference"] for item in records.values()]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(reference, prediction)["rougeL"].fmeasure
                  for prediction, reference in zip(predictions, references)) / len(predictions)
    metrics = {"dataset": dataset, "num_samples": len(records),
               "empty_predictions": sum(not item.strip() for item in raw_predictions),
               "rougeL_percent": rouge_l * 100,
               "sacre_bleu": corpus_bleu(predictions, [references]).score if dataset == "IWSLT" else ""}
    target.with_name("primary_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"policy_id": policy, "dataset": dataset, "samples": metrics["num_samples"],
            "rougeL_percent": metrics.get("rougeL_percent", ""),
            "bert_score_percent": metrics.get("bert_score_percent", ""),
            "sacre_bleu": metrics.get("sacre_bleu", ""),
            "empty_predictions": metrics["empty_predictions"]}


def main() -> None:
    rows = [merge(policy, dataset, expected)
            for policy in ("p03", "p04")
            for dataset, expected in DATASETS.items()]
    output = EXP / "task_quality/report/uniform_p03_p04_backfill_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
