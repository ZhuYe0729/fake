#!/usr/bin/env python3
"""Build paper-facing tables and measured-speed task Pareto plots for 057."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EXP = Path(os.environ["COSPAQ_EXPERIMENT_DIR"])
TASK = EXP / "task_quality"
METRICS = (
    ("cnn_dm_1000", "rougeL_percent", "CNN/DM ROUGE-L (%)", "cnn_rougel"),
    ("cnn_dm_1000", "bert_score_percent", "CNN/DM BERTScore (%)", "cnn_bertscore"),
    ("dsum", "rougeL_percent", "DialogSum ROUGE-L (%)", "dsum_rougel"),
    ("dsum", "bert_score_percent", "DialogSum BERTScore (%)", "dsum_bertscore"),
    ("IWSLT", "sacre_bleu", "IWSLT SacreBLEU", "iwslt_bleu"),
)
UNIFORM_NAMES = {
    "p00": "dense BF16", "p01": "dense NVFP4", "p02": "sparse BF16",
    "p03": "sparse NVFP4 projection", "p04": "W4A16/Marlin",
}
LABELLED_OURS = {"point_000", "point_001", "point_005", "point_006", "point_008", "point_009", "point_011"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def frontier(items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Non-dominated measured points, maximizing both speed and task score."""
    ordered = sorted(items, key=lambda item: float(item["speedup"]))
    best = float("-inf"); result = []
    for item in ordered:
        score = float(item["score"])
        if score > best + 1e-10:
            result.append(item); best = score
    return result


def main() -> None:
    task_rows = read(TASK / "summary.csv")
    closure = {row["policy_id"]: row for row in read(EXP / "pareto/closure_summary.csv")}
    uniform_speed = {row["policy_id"]: row for row in read(EXP / "speed/uniform_baselines.csv")}
    predicted = {row["policy_id"]: row for row in read(EXP / "pareto/predicted_points.csv")}
    report = TASK / "report"; report.mkdir(exist_ok=True)

    by_policy: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    kinds: dict[str, str] = {}
    for row in task_rows:
        by_policy[row["policy_id"]][row["dataset"]] = row; kinds[row["policy_id"]] = row["kind"]

    records = []
    for policy_id in sorted(by_policy, key=lambda item: (kinds[item] != "uniform", item)):
        rows = by_policy[policy_id]; kind = kinds[policy_id]
        speedup = (float(uniform_speed[policy_id]["speedup_vs_dense_bf16"])
                   if kind == "uniform" else float(closure[policy_id]["measured_speedup_vs_dense"]))
        record = {"policy_id": policy_id, "display_name": UNIFORM_NAMES.get(policy_id, policy_id),
                  "family": kind, "measured_speedup_vs_dense_bf16": speedup,
                  "predicted_delta_nll": float(predicted[policy_id]["predicted_delta_nll"]) if policy_id in predicted else ""}
        for dataset, metric, _, short in METRICS:
            record[short] = float(rows[dataset][metric])
        records.append(record)

    fields = list(records[0])
    with (report / "all_policy_task_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)

    for dataset, metric, ylabel, short in METRICS:
        data = [{"policy_id": row["policy_id"], "name": row["display_name"], "family": row["family"],
                 "speedup": row["measured_speedup_vs_dense_bf16"], "score": row[short]} for row in records]
        uniform = [item for item in data if item["family"] == "uniform"]
        ours = [item for item in data if item["family"] == "ours"]
        ours_frontier = frontier([item for item in ours if item["policy_id"] != "point_007"])
        fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
        ax.scatter([item["speedup"] for item in uniform], [item["score"] for item in uniform],
                   marker="s", s=105, color="#D62728", zorder=3, label="Uniform baselines")
        ax.scatter([item["speedup"] for item in ours], [item["score"] for item in ours],
                   marker="o", s=88, color="#1F2A3A", zorder=3, label="Ours (measured)")
        if len(ours_frontier) > 1:
            ax.plot([item["speedup"] for item in ours_frontier], [item["score"] for item in ours_frontier],
                    color="#1F2A3A", lw=2.6, alpha=.9, zorder=2, label="Ours non-dominated envelope")
        for item in uniform:
            offset = (-58, 10) if item["policy_id"] == "p00" else (5, 5)
            ax.annotate(item["name"], (item["speedup"], item["score"]), xytext=offset,
                        textcoords="offset points", fontsize=8.5, color="#A51D1D")
        for item in ours:
            if item["policy_id"] not in LABELLED_OURS: continue
            offset = (7, 13) if item["policy_id"] == "point_000" else (5, 5)
            ax.annotate(item["policy_id"], (item["speedup"], item["score"]), xytext=offset,
                        textcoords="offset points", fontsize=8)
        ax.set_title(f"Llama-3.1-8B-Instruct prefill-decode: speed vs {ylabel}")
        ax.set_xlabel("Measured E2E speedup vs dense BF16")
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(frameon=True, fontsize=9)
        fig.savefig(report / f"pareto_{short}.png", dpi=260); plt.close(fig)

    lines = ["# Llama-3.1-8B-Instruct: prefill-decode task validation", "",
             "Protocol: B=8, input=2048, output=64; VLLM V1 `phase_hetero_mytest`; BF16 KV; chunked prefill disabled. All task scores use real phase-runtime generation. Speed is fresh-process measured E2E speed under the common 0.80 GPU-memory protocol.", "",
             "`point_007` is retained as a measured data point but excluded from the line envelope because its speed measurement is anomalously below dense BF16 (0.739x).", "",
             "Recommended paper candidates: `point_005` is the quality-preserving speed point (1.262x); `point_006` is the stronger-speed trade-off point (1.321x).", "",
             "| policy | family | measured speedup | predicted ΔNLL | CNN R-L | CNN BERT | DialogSum R-L | DialogSum BERT | IWSLT BLEU |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        nll = "" if row["predicted_delta_nll"] == "" else f"{float(row['predicted_delta_nll']):.4f}"
        lines.append(f"| {row['display_name']} | {row['family']} | {row['measured_speedup_vs_dense_bf16']:.3f} | {nll} | {row['cnn_rougel']:.3f} | {row['cnn_bertscore']:.3f} | {row['dsum_rougel']:.3f} | {row['dsum_bertscore']:.3f} | {row['iwslt_bleu']:.3f} |")
    (report / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
