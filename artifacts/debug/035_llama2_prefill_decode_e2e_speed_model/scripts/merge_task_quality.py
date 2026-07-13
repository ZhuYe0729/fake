#!/usr/bin/env python3
"""Merge PMPD quality shards, compute metrics, and write the Pareto summary."""
from __future__ import annotations

import csv
import json
import subprocess
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
ALL_POINTS = tuple(range(12))
PYTHON = "/home/agent/wja/miniconda3/envs/vllm/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="task_quality_continuous")
    parser.add_argument("--points", default=",".join(map(str, ALL_POINTS)))
    args = parser.parse_args()
    quality_root = ROOT / args.output_name
    points = tuple(int(item) for item in args.points.split(",") if item.strip())
    if not points or any(point not in ALL_POINTS for point in points):
        raise ValueError(f"--points must be drawn from {ALL_POINTS}")
    speed = {int(row["point"]): row for row in csv.DictReader((ROOT / "report/formal_util085_actual_nll_summary.csv").open())}
    rows = []
    for point in points:
        label = f"ours_point_{point}_prefill_decode"
        for dataset, expected in DATASETS.items():
            records = {}
            for path in sorted((quality_root / "shards" / f"point_{point}" / dataset).glob("shard_*/**/*.jsonl")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line); key = str(item["question_id"])
                    if key in records:
                        raise RuntimeError(f"duplicate {dataset} point={point} question_id={key}")
                    records[key] = item
            if len(records) != expected:
                raise RuntimeError(f"incomplete {dataset} point={point}: {len(records)} != {expected}")
            target = quality_root / "results" / f"point_{point}" / dataset / f"{label}-fp16.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            ordered = sorted(records.values(), key=lambda item: int(item["question_id"]) if str(item["question_id"]).isdigit() else str(item["question_id"]))
            target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")
            subprocess.run([PYTHON, str(REPO / "references/pmpd_eval_kit/pmpd_eval.py"), "--dataset", dataset,
                            "--metrics-only", str(target)], check=True)
            metrics = json.loads(target.with_name("metrics.json").read_text())
            s = speed[point]
            rows.append({"point": point, "dataset": dataset, "samples": metrics["num_samples"],
                         "e2e_median_ms": s["e2e_median_ms"], "speedup_vs_dense": s["speedup_vs_point0"],
                         "wikitext_delta_nll": s["measured_wikitext_delta_nll"],
                         "rougeL_percent": metrics.get("rougeL_percent", ""),
                         "bert_score_percent": metrics.get("bert_score_percent", ""),
                         "sacre_bleu": metrics.get("sacre_bleu", ""),
                         "empty_predictions": metrics["empty_predictions"]})
    out = quality_root / "summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Prefill-decode PMPD task-quality validation", "", "All speed values use the `.85` formal protocol; task metrics use isolated fresh-process vLLM generation.", "",
             "| point | dataset | speedup | WikiText ΔNLL | ROUGE-L | BERTScore | SacreBLEU | empty |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        f = lambda key: "" if row[key] == "" else f"{float(row[key]):.3f}"
        lines.append(f"| {row['point']} | {row['dataset']} | {float(row['speedup_vs_dense']):.3f} | {float(row['wikitext_delta_nll']):.3f} | {f('rougeL_percent')} | {f('bert_score_percent')} | {f('sacre_bleu')} | {row['empty_predictions']} |")
    (quality_root / "summary.md").write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
