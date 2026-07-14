#!/usr/bin/env python3
"""Plot real-task Pareto curves with old and stall-screened intermediate points."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
OLD = REPO / "artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/task_quality_all/summary.csv"
BASELINES = REPO / "artifacts/exports/vllm/baselines/llama2-7b-chat/results/summary/quality_summary.csv"
BASE_SPEED = REPO / "artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/baseline_speed_util085/summary.csv"
NEW = ROOT / "task_quality_intermediate/summary.csv"
METRICS = {"cnn_dm_1000": ("rougeL_percent", "CNN/DM ROUGE-L"), "dsum": ("rougeL_percent", "DialogSum ROUGE-L"), "IWSLT": ("sacre_bleu", "IWSLT SacreBLEU")}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    base_speed = [row for row in read(BASE_SPEED) if row["scenario"] == "prefill_decode"]
    dense_ms = float(next(row["e2e_median_ms"] for row in base_speed if row["method"] == "dense_bf16"))
    base_x = {row["method"]: dense_ms / float(row["e2e_median_ms"]) for row in base_speed}
    data = []
    for row in read(OLD):
        metric, _ = METRICS[row["dataset"]]
        data.append({"family": "old", "label": f"point_{row['point']}", "dataset": row["dataset"], "speedup": float(row["speedup_vs_dense"]), "score": float(row[metric]), "screened": False})
    for row in read(NEW):
        metric, _ = METRICS[row["dataset"]]
        data.append({"family": "intermediate", "label": f"i{row['point']}", "dataset": row["dataset"], "speedup": float(row["screened_speedup_vs_dense"]), "score": float(row[metric]), "screened": True})
    for row in read(BASELINES):
        metric, _ = METRICS[row["dataset"]]
        data.append({"family": "baseline", "label": row["method"], "dataset": row["dataset"], "speedup": base_x[row["method"]], "score": float(row[metric]), "screened": False})
    out = ROOT / "task_quality_intermediate/report"; out.mkdir(parents=True, exist_ok=True)
    with (out / "all_task_pareto_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    labels = {"dense_bf16": "dense BF16", "dense_nvfp4": "dense NVFP4", "marlin_nvfp4": "Marlin NVFP4", "sparse_bf16": "sparse BF16", "sparse_nvfp4": "sparse NVFP4"}
    for dataset, (_, metric_name) in METRICS.items():
        subset = [row for row in data if row["dataset"] == dataset]
        old = [row for row in subset if row["family"] == "old" and row["label"] != "point_9"]
        intermediate = [row for row in subset if row["family"] == "intermediate"]
        candidates = old + intermediate
        frontier = [row for row in candidates if not any(other is not row and other["speedup"] >= row["speedup"] and other["score"] >= row["score"] and (other["speedup"] > row["speedup"] or other["score"] > row["score"]) for other in candidates)]
        frontier.sort(key=lambda row: row["speedup"])
        dominated = [row for row in candidates if row not in frontier]
        baselines = [row for row in subset if row["family"] == "baseline"]
        fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
        ax.plot([r["speedup"] for r in frontier], [r["score"] for r in frontier], "-o", color="#202B3C", linewidth=3, markersize=8, label="Ours frontier")
        if dominated:
            ax.scatter([r["speedup"] for r in dominated], [r["score"] for r in dominated], marker="x", s=85, color="#8292A8", label="Ours dominated")
        ax.scatter([r["speedup"] for r in intermediate], [r["score"] for r in intermediate], marker="D", s=75, color="#0F766E", label="New intermediate (stall-screened)", zorder=5)
        ax.scatter([r["speedup"] for r in baselines], [r["score"] for r in baselines], marker="s", s=125, color="#D62728", label="Uniform baselines", zorder=4)
        for row in intermediate:
            ax.annotate(row["label"], (row["speedup"], row["score"]), xytext=(6, 7), textcoords="offset points", color="#0F766E", fontsize=10)
        for row in baselines:
            ax.annotate(labels[row["label"]], (row["speedup"], row["score"]), xytext=(6, 7), textcoords="offset points", color="#B51F24", fontsize=9)
        ax.set_title(f"Llama2-7B prefill-decode: speedup vs {metric_name}")
        ax.set_xlabel("E2E speedup vs dense BF16 (higher is better)")
        ax.set_ylabel(f"Actual task score: {metric_name} (higher is better)")
        ax.grid(alpha=.28); ax.margins(x=.07, y=.12); ax.legend(loc="best")
        fig.savefig(out / f"pareto_{dataset}.png", dpi=260)


if __name__ == "__main__":
    main()
