#!/usr/bin/env python3
"""Render three real-task Pareto plots: all ours points plus uniform baselines."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT.parents[2] / "artifacts/exports/vllm/baselines/llama2-7b-chat"
DATASET_METRIC = {"cnn_dm_1000": ("rougeL_percent", "CNN/DM ROUGE-L"),
                  "dsum": ("rougeL_percent", "DialogSum ROUGE-L"),
                  "IWSLT": ("sacre_bleu", "IWSLT SacreBLEU")}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", default="task_quality_all")
    args = parser.parse_args()
    quality_root = ROOT / args.quality_root
    ours = rows(quality_root / "summary.csv")
    baseline_quality = rows(BASELINE_ROOT / "results/summary/quality_summary.csv")
    baseline_speed = [row for row in rows(ROOT / "baseline_speed_util085/summary.csv") if row["scenario"] == "prefill_decode"]
    dense_ms = float(next(row["e2e_median_ms"] for row in baseline_speed if row["method"] == "dense_bf16"))
    base_speedup = {row["method"]: dense_ms / float(row["e2e_median_ms"]) for row in baseline_speed}
    combined = []
    for row in ours:
        metric, _ = DATASET_METRIC[row["dataset"]]
        combined.append({"family": "ours", "label": f"point_{row['point']}", "dataset": row["dataset"],
                         "speedup": float(row["speedup_vs_dense"]), "score": float(row[metric]),
                         "unstable_speed": row["point"] == "9"})
    for row in baseline_quality:
        metric, _ = DATASET_METRIC[row["dataset"]]
        combined.append({"family": "baseline", "label": row["method"], "dataset": row["dataset"],
                         "speedup": base_speedup[row["method"]], "score": float(row[metric]),
                         "unstable_speed": False})
    out = quality_root / "report"; out.mkdir(exist_ok=True)
    with (out / "all_task_pareto_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0])); writer.writeheader(); writer.writerows(combined)
    labels = {"dense_bf16": "dense BF16", "dense_nvfp4": "dense NVFP4", "marlin_nvfp4": "Marlin NVFP4",
              "sparse_bf16": "sparse BF16", "sparse_nvfp4": "sparse NVFP4"}
    for dataset, (metric, title_metric) in DATASET_METRIC.items():
        data = [row for row in combined if row["dataset"] == dataset]
        ours_data = [row for row in data if row["family"] == "ours"]
        stable = [row for row in ours_data if not row["unstable_speed"]]
        stable.sort(key=lambda row: row["speedup"])
        frontier = [row for row in stable if not any(other is not row and other["speedup"] >= row["speedup"] and other["score"] >= row["score"] and (other["speedup"] > row["speedup"] or other["score"] > row["score"]) for other in stable)]
        frontier.sort(key=lambda row: row["speedup"])
        dominated = [row for row in stable if row not in frontier]
        baselines = [row for row in data if row["family"] == "baseline"]
        unstable = [row for row in ours_data if row["unstable_speed"]]
        fig, ax = plt.subplots(figsize=(10.6, 6.1), constrained_layout=True)
        ax.plot([row["speedup"] for row in frontier], [row["score"] for row in frontier], "-o", color="#202B3C", linewidth=3, markersize=9, label="Ours measured frontier", zorder=3)
        if dominated:
            ax.scatter([row["speedup"] for row in dominated], [row["score"] for row in dominated], marker="x", s=90, linewidth=2, color="#8292A8", label="Ours dominated", zorder=2)
        ax.scatter([row["speedup"] for row in baselines], [row["score"] for row in baselines], marker="s", s=135, color="#D62728", label="Uniform baselines", zorder=4)
        for row in baselines:
            ax.annotate(labels[row["label"]], (row["speedup"], row["score"]), xytext=(7, 7), textcoords="offset points", color="#B51F24", fontsize=10)
        for row in unstable:
            ax.scatter(row["speedup"], row["score"], marker="X", s=150, color="#8E44AD", zorder=5, label="point 9 (unstable speed)")
        max_speed = next(row for row in ours_data if row["label"] == "point_11")
        ax.scatter(max_speed["speedup"], max_speed["score"], marker="*", s=300, color="#F0A202", edgecolor="#202B3C", linewidth=1.2, zorder=6, label="Ours max-speed")
        ax.set_title(f"Llama2-7B prefill-decode: speedup vs {title_metric}")
        ax.set_xlabel("Measured E2E speedup vs dense BF16 (higher is better)")
        ax.set_ylabel(f"Actual task score: {title_metric} (higher is better)")
        ax.grid(alpha=.28); ax.margins(x=.07, y=.12); ax.legend(loc="best", frameon=True)
        fig.savefig(out / f"pareto_{dataset}_{metric}.png", dpi=260)
        plt.close(fig)


if __name__ == "__main__":
    main()
