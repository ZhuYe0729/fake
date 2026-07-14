#!/usr/bin/env python3
"""Audit, deduplicate identical shard records, and score intermediate policies."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
POINTS = (34, 36, 37, 38)
SPEED = ROOT / "report/intermediate_actual_nll_summary.csv"


def normalized(text: str) -> str:
    return text if text.strip() else "."


def main() -> None:
    speed = {int(row["source_point"]): row for row in csv.DictReader(SPEED.open())}
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = []
    audit = []
    root = ROOT / "task_quality_intermediate"
    for point in POINTS:
        label = f"ours_intermediate_{point}_prefill_decode"
        for dataset, expected in DATASETS.items():
            records: dict[str, dict] = {}
            duplicate = 0
            for path in sorted((root / "shards" / f"point_{point}" / dataset).glob("shard_*/**/*.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    key = str(item["question_id"])
                    if key in records:
                        duplicate += 1
                        if records[key] != item:
                            raise RuntimeError(f"non-identical duplicate point={point} dataset={dataset} id={key}")
                    else:
                        records[key] = item
            if len(records) != expected:
                raise RuntimeError(f"incomplete point={point} dataset={dataset}: {len(records)} != {expected}")
            ordered = sorted(records.values(), key=lambda item: int(item["question_id"]))
            predictions = [normalized(item["choices"][0]["turns"][0]) for item in ordered]
            references = [item["reference"] for item in ordered]
            empty = sum(not item["choices"][0]["turns"][0].strip() for item in ordered)
            rouge_l = "" if dataset == "IWSLT" else 100 * sum(rouge.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)) / expected
            bleu = corpus_bleu(predictions, [references]).score if dataset == "IWSLT" else ""
            s = speed[point]
            rows.append({"point": point, "dataset": dataset, "samples": expected,
                         "screened_speedup_vs_dense": s["speedup_vs_point0"], "screened_e2e_median_ms": s["e2e_median_ms"],
                         "wikitext_delta_nll": s["measured_wikitext_delta_nll"], "rougeL_percent": rouge_l,
                         "sacre_bleu": bleu, "empty_predictions": empty})
            audit.append({"point": point, "dataset": dataset, "unique_records": expected, "identical_duplicates_removed": duplicate})
    for name, data in (("summary.csv", rows), ("audit.csv", audit)):
        with (root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    print(root / "summary.csv")
    print(root / "audit.csv")


if __name__ == "__main__":
    main()
