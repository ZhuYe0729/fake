#!/usr/bin/env python3
"""Merge PMPD shards, enforce exact question coverage and compute all metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pmpd_eval
from common import BERTSCORE_MODEL, IWSLT_FILTER_TOKENIZER, PMPD, RUN, write_json


def main() -> None:
    selected = json.loads((RUN / "tasks/selection.json").read_text())["selected"]
    summary = []
    for label in selected:
        for dataset, expected in PMPD["datasets"].items():
            records = []
            pattern = RUN / f"tasks/shards/{label}/{dataset}"
            for path in sorted(pattern.glob(f"*/*/{label}-fp16.jsonl")):
                records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
            ids = [row["question_id"] for row in records]
            if len(records) != expected or len(set(ids)) != expected:
                raise RuntimeError(f"{label}/{dataset}: expected {expected} unique rows, got {len(records)}/{len(set(ids))}")
            records.sort(key=lambda row: row["question_id"])
            output = RUN / f"tasks/merged/{label}/{dataset}/{label}-fp16.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            args = argparse.Namespace(dataset=dataset, split="test", bertscore_model=BERTSCORE_MODEL,
                                      bertscore_num_layers=17,
                                      iwslt_filter_tokenizer=str(IWSLT_FILTER_TOKENIZER))
            metrics_path = pmpd_eval.compute_metrics(args, output)
            metrics = json.loads(metrics_path.read_text())
            write_json(output.parent / f"{label}_metrics.json", metrics)
            summary.append({"label": label, **metrics})
    write_json(RUN / "tasks/summary.json", {"rows": summary, "selected": selected,
                                             "expected_rows": len(selected) * 3})
    print(json.dumps({"metric_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
