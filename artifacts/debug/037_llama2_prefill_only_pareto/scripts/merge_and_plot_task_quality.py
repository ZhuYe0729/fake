#!/usr/bin/env python3
"""Merge complete task shards, compute PMPD metrics, and plot task Pareto curves."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
POINTS = (6, 9, 15)
DATASETS = {"cnn_dm_1000": 1000, "dsum": 1500, "IWSLT": 333}
METRICS = {"cnn_dm_1000": ("rougeL_percent", "CNN/DM ROUGE-L"), "dsum": ("rougeL_percent", "DialogSum ROUGE-L"), "IWSLT": ("sacre_bleu", "IWSLT SacreBLEU")}
PYTHON = "/home/agent/wja/miniconda3/envs/vllm/bin/python"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def merge(point: int, dataset: str, expected: int) -> Path:
    records: dict[str, dict] = {}
    for path in sorted((ROOT / "task_quality/shards" / f"point_{point}" / dataset).glob("shard_*/**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            key = str(item["question_id"])
            if key in records:
                raise RuntimeError(f"duplicate point={point} dataset={dataset} question_id={key}")
            records[key] = item
    if len(records) != expected:
        raise RuntimeError(f"incomplete point={point} dataset={dataset}: {len(records)} != {expected}")
    label = f"ours_point_{point}_prefill_only"
    target = ROOT / "task_quality/results" / f"point_{point}" / dataset / f"{label}-fp16.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda item: int(item["question_id"]) if str(item["question_id"]).isdigit() else str(item["question_id"]))
    target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")
    return target


def frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    kept = [row for row in rows if not any(other is not row and float(other["speedup"]) >= float(row["speedup"]) and float(other["score"]) >= float(row["score"]) and (float(other["speedup"]) > float(row["speedup"]) or float(other["score"]) > float(row["score"])) for other in rows)]
    return sorted(kept, key=lambda row: float(row["speedup"]))


def main() -> None:
    speed_rows = {int(row["point_index"]): row for row in read(ROOT / "report/actual_nll_speed_summary.csv") if row["family"] == "ours"}
    summary: list[dict[str, object]] = []
    for point in POINTS:
        for dataset, expected in DATASETS.items():
            target = merge(point, dataset, expected)
            subprocess.run([PYTHON, str(REPO / "references/pmpd_eval_kit/pmpd_eval.py"), "--dataset", dataset, "--metrics-only", str(target)], check=True)
            metrics = json.loads(target.with_name("metrics.json").read_text())
            speed = speed_rows[point]
            summary.append({"point": point, "dataset": dataset, "samples": metrics["num_samples"],
                            "empty_predictions": metrics["empty_predictions"], "speedup_vs_dense": speed["speedup_vs_dense"],
                            "e2e_median_ms": speed["e2e_median_ms"], "wikitext_delta_nll": speed["raw_delta_nll"],
                            "rougeL_percent": metrics.get("rougeL_percent", ""), "bert_score_percent": metrics.get("bert_score_percent", ""),
                            "sacre_bleu": metrics.get("sacre_bleu", "")})
    out = ROOT / "task_quality"
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)

    baseline_quality = read(REPO / "artifacts/exports/vllm/baselines/llama2-7b-chat/results/summary/quality_summary.csv")
    baseline_speed = {row["label"]: row for row in read(ROOT / "report/actual_nll_speed_summary.csv") if row["family"] == "uniform"}
    report = out / "report"; report.mkdir(exist_ok=True)
    all_rows = []
    for dataset, (metric, title) in METRICS.items():
        ours = [{"family": "ours", "label": f"ours {r['point']}", "dataset": dataset, "speedup": float(r["speedup_vs_dense"]), "score": float(r[metric])} for r in summary if r["dataset"] == dataset]
        bases = [{"family": "uniform", "label": r["method"], "dataset": dataset, "speedup": float(baseline_speed[r["method"]]["speedup_vs_dense"]), "score": float(r[metric])} for r in baseline_quality if r["dataset"] == dataset]
        all_rows.extend(ours + bases)
        ours_front = frontier(ours)
        fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
        ax.plot([r["speedup"] for r in ours_front], [r["score"] for r in ours_front], "-o", color="#202B3C", linewidth=3, markersize=8, label="Ours frontier")
        ax.scatter([r["speedup"] for r in ours], [r["score"] for r in ours], marker="o", s=75, color="#64748B", label="Ours measured")
        ax.scatter([r["speedup"] for r in bases], [r["score"] for r in bases], marker="s", s=125, color="#D62728", label="Uniform baselines", zorder=4)
        for r in ours:
            ax.annotate(r["label"], (r["speedup"], r["score"]), xytext=(6, 7), textcoords="offset points", fontsize=10)
        for r in bases:
            ax.annotate(r["label"].replace("_", " "), (r["speedup"], r["score"]), xytext=(6, 7), textcoords="offset points", color="#B51F24", fontsize=9)
        ax.set_title(f"Llama2-7B prefill-only: speedup vs {title}")
        ax.set_xlabel("Measured E2E prefill speedup vs dense BF16")
        ax.set_ylabel(f"Actual task score: {title}")
        ax.grid(alpha=.28); ax.margins(x=.07, y=.14); ax.legend(loc="best")
        fig.savefig(report / f"pareto_{dataset}.png", dpi=240)
    with (report / "all_task_pareto_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    print(out / "summary.csv")


if __name__ == "__main__":
    main()
