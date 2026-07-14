#!/usr/bin/env python3
"""Create task-level Pareto summaries from measured Llama-3.1 artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[2] / "artifacts/exports/vllm/baselines/llama3.1-8b-instruct/results"
MAX_SPEED = ROOT.parents[2] / "artifacts/exports/vllm/ours/llama3.1-8b-instruct/max_speed/prefill_decode/results/quality"
DATASETS = {
    "cnn_dm_1000": ("ROUGE-L", "rougeL_percent"),
    "dsum": ("ROUGE-L", "rougeL_percent"),
    "IWSLT": ("SacreBLEU", "sacre_bleu"),
}
DISPLAY = {"point_002": "ours-high-Q", "point_004": "ours-mid", "point_009_max_speed": "ours-max"}


def metric(path: Path, key: str) -> float:
    return float(json.loads(path.read_text())[key])


def closure_speeds() -> dict[str, float]:
    rows = csv.DictReader((ROOT / "closure/summary.csv").open())
    return {row["policy_id"]: float(row["speedup_vs_dense"]) for row in rows}


def main() -> None:
    speeds = closure_speeds()
    out = ROOT / "closure/tasks/report"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for dataset, (metric_name, metric_key) in DATASETS.items():
        for method in ("dense_bf16", "dense_nvfp4", "sparse_bf16"):
            path = BASE / "quality" / method / dataset / "metrics.json"
            speed_key = {"dense_bf16": "dense_bf16", "dense_nvfp4": "dense_nvfp4", "sparse_bf16": "sparse_bf16"}[method]
            rows.append({"dataset": dataset, "metric": metric_name, "kind": "uniform",
                         "policy": method, "speedup_vs_dense": speeds[speed_key],
                         "score": metric(path, metric_key), "metrics_path": str(path)})
        for point in ("point_002", "point_004"):
            path = ROOT / "closure/tasks" / point / "results/quality" / dataset / "metrics.json"
            if path.exists():
                rows.append({"dataset": dataset, "metric": metric_name, "kind": "ours",
                             "policy": point, "speedup_vs_dense": speeds[point],
                             "score": metric(path, metric_key), "metrics_path": str(path)})
        path = MAX_SPEED / dataset / "metrics.json"
        rows.append({"dataset": dataset, "metric": metric_name, "kind": "ours",
                     "policy": "point_009_max_speed", "speedup_vs_dense": speeds["point_009"],
                     "score": metric(path, metric_key), "metrics_path": str(path)})

    fields = ["dataset", "metric", "kind", "policy", "speedup_vs_dense", "score", "metrics_path"]
    with (out / "downstream_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Llama-3.1-8B-Instruct prefill-decode downstream Pareto", "",
             "All horizontal coordinates are freshly measured continuous phase-heterogeneous E2E speedups from `closure/summary.csv`. "
             "Uniform task quality is read-only from the frozen baseline artifacts; point_009 uses the pre-existing max-speed task run.", ""]
    for dataset, (metric_name, _) in DATASETS.items():
        lines.extend([f"## {dataset} ({metric_name})", "",
                      "| kind | policy | speedup vs dense | score |", "|---|---|---:|---:|"])
        subset = [row for row in rows if row["dataset"] == dataset]
        for row in sorted(subset, key=lambda item: float(item["speedup_vs_dense"])):
            lines.append(f"| {row['kind']} | {row['policy']} | {float(row['speedup_vs_dense']):.3f} | {float(row['score']):.3f} |")
        lines.append("")
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        for kind, color, marker, label in (("uniform", "#d62728", "s", "Uniform baselines"),
                                           ("ours", "#1f2937", "o", "Ours: Pareto policies")):
            points = [row for row in subset if row["kind"] == kind]
            points.sort(key=lambda item: float(item["speedup_vs_dense"]))
            ax.plot([row["speedup_vs_dense"] for row in points], [row["score"] for row in points],
                    color=color, marker=marker, linewidth=2.6, markersize=8, label=label)
            for row in points:
                ax.annotate(DISPLAY.get(str(row["policy"]), str(row["policy"])),
                            (row["speedup_vs_dense"], row["score"]),
                            xytext=(5, 6), textcoords="offset points", fontsize=9, color=color)
        ax.set_title(f"Llama-3.1-8B-Instruct PMPD: speed vs {metric_name} ({dataset})", fontsize=13)
        ax.set_xlabel("Measured E2E speedup vs dense BF16")
        ax.set_ylabel(metric_name)
        xs = [float(row["speedup_vs_dense"]) for row in subset]
        ax.set_xlim(min(xs) - 0.035, max(xs) + 0.17)
        ax.grid(alpha=0.25); ax.legend(loc="best")
        fig.savefig(out / f"pareto_speed_vs_{dataset}.png", dpi=190)
        plt.close(fig)
    (out / "downstream_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
