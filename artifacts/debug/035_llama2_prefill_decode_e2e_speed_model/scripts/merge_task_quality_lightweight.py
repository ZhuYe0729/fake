#!/usr/bin/env python3
"""Merge completed PMPD shards with the task metrics used in Pareto plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}


def normalized(text: str) -> str:
    return text if text.strip() else "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--points", required=True)
    args = parser.parse_args()
    points = tuple(int(item) for item in args.points.split(",") if item.strip())
    root = ROOT / args.output_name
    speed = {int(row["point"]): row for row in csv.DictReader((ROOT / "report/formal_util085_actual_nll_summary.csv").open())}
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = []
    for point in points:
        for dataset, expected in DATASETS.items():
            records = {}
            for path in (root / "shards" / f"point_{point}" / dataset).glob("shard_*/**/*.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    key = str(record["question_id"])
                    if key in records:
                        raise RuntimeError(f"duplicate {dataset} point={point} id={key}")
                    records[key] = record
            if len(records) != expected:
                raise RuntimeError(f"incomplete {dataset} point={point}: {len(records)} != {expected}")
            ordered = sorted(records.values(), key=lambda item: int(item["question_id"]))
            predictions = [normalized(item["choices"][0]["turns"][0]) for item in ordered]
            references = [item["reference"] for item in ordered]
            empty = sum(not item["choices"][0]["turns"][0].strip() for item in ordered)
            metrics = {"rougeL_percent": "", "sacre_bleu": ""}
            if dataset == "IWSLT":
                metrics["sacre_bleu"] = corpus_bleu(predictions, [references]).score
            else:
                metrics["rougeL_percent"] = 100 * sum(
                    rouge.score(reference, prediction)["rougeL"].fmeasure
                    for prediction, reference in zip(predictions, references)
                ) / expected
            item = speed[point]
            rows.append({"point": point, "dataset": dataset, "samples": expected,
                         "e2e_median_ms": item["e2e_median_ms"], "speedup_vs_dense": item["speedup_vs_point0"],
                         "wikitext_delta_nll": item["measured_wikitext_delta_nll"],
                         "rougeL_percent": metrics["rougeL_percent"], "bert_score_percent": "",
                         "sacre_bleu": metrics["sacre_bleu"], "empty_predictions": empty})
    out = root / "summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
