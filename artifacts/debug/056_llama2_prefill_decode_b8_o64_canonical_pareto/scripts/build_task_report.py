#!/usr/bin/env python3
"""Write task tables and clearly-labelled task-quality Pareto visualizations."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat"))
EXP = EXPERIMENT / "task_quality"
BASE = Path(os.environ.get("COSPAQ_BASELINE_DIR", ROOT / "artifacts/exports/vllm/baselines/llama2-7b-chat"))
METRICS = {"cnn_dm_1000": ("rougeL_percent", "CNN/DM ROUGE-L (%)"),
           "dsum": ("rougeL_percent", "DialogSum ROUGE-L (%)"),
           "IWSLT": ("sacre_bleu", "IWSLT SacreBLEU")}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def main() -> None:
    ours = rows(EXP / "summary.csv")
    quality = rows(BASE / "results/summary/quality_summary.csv")
    speed = rows(EXPERIMENT / "speed/uniform_baselines.csv")
    dense = float(next(row["e2e_median_ms"] for row in speed if row["method"] == "dense_bf16"))
    speedup = {row["method"]: dense / float(row["e2e_median_ms"]) for row in speed}
    report = EXP / "report"; report.mkdir(exist_ok=True)
    combined = []
    for row in ours:
        actual = row["measured_speedup_vs_dense"]
        combined.append({"family": "ours", "label": row["policy_id"], "dataset": row["dataset"],
                         "score": row[METRICS[row["dataset"]][0]], "speedup": actual or row["raw_predicted_speedup_vs_dense"],
                         "speed_kind": "measured" if actual else "roofline_predicted"})
    for row in quality:
        if row["method"] not in speedup: continue
        combined.append({"family": "uniform", "label": row["method"], "dataset": row["dataset"],
                         "score": row[METRICS[row["dataset"]][0]], "speedup": str(speedup[row["method"]]),
                         "speed_kind": "baseline_measured"})
    with (report / "task_pareto_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0])); writer.writeheader(); writer.writerows(combined)
    for dataset, (_, ylabel) in METRICS.items():
        data = [row for row in combined if row["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
        uniform = [row for row in data if row["family"] == "uniform"]
        ours_measured = [row for row in data if row["family"] == "ours" and row["speed_kind"] == "measured"]
        ours_predicted = [row for row in data if row["family"] == "ours" and row["speed_kind"] != "measured"]
        ax.scatter([float(x["speedup"]) for x in uniform], [float(x["score"]) for x in uniform], marker="s", s=115, color="#D62728", label="Uniform baselines")
        ax.scatter([float(x["speedup"]) for x in ours_measured], [float(x["score"]) for x in ours_measured], marker="o", s=120, color="#202B3C", label="Ours (measured speed)")
        ax.scatter([float(x["speedup"]) for x in ours_predicted], [float(x["score"]) for x in ours_predicted], marker="o", s=100, facecolors="none", edgecolors="#202B3C", linewidth=2, label="Ours (roofline speed prediction)")
        for row in data:
            ax.annotate(row["label"], (float(row["speedup"]), float(row["score"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(f"{os.environ.get('COSPAQ_MODEL_LABEL', 'Llama2-7B')} prefill-decode: speed vs {ylabel}")
        ax.set_xlabel("E2E speedup vs dense BF16"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(frameon=True)
        fig.savefig(report / f"pareto_{dataset}.png", dpi=240); plt.close(fig)
    lines = ["# Canonical prefill-decode downstream-task validation", "",
             "All listed Pareto policies were generated with the canonical sparse phase runtime. Solid ours markers use fresh-process measured speed under the common gpu_memory_utilization=0.80 protocol; hollow ours markers use only the roofline screening speed and are not final speed claims.", "",
             "| policy | dataset | task score | measured ΔNLL | speed source | speedup |", "|---|---|---:|---:|---|---:|"]
    for row in ours:
        metric = METRICS[row["dataset"]][0]; measured = row["measured_speedup_vs_dense"]
        lines.append(f"| {row['policy_id']} | {row['dataset']} | {float(row[metric]):.3f} | {row['measured_delta_nll'] or ''} | {'measured' if measured else 'roofline predicted'} | {float(measured or row['raw_predicted_speedup_vs_dense']):.3f} |")
    (report / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
