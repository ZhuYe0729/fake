#!/usr/bin/env python3
"""Build the paper-facing Llama3 prefill-only task/latency Pareto report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from scenario import EXP

UNIFORM = {
    "p00": "dense BF16",
    "p01": "dense NVFP4",
    "p02": "sparse BF16",
    "p03": "sparse NVFP4",
    "p04": "Marlin W4A16",
}
TASKS = {
    "wikitext": ("word_perplexity,none", "WikiText word PPL", False),
    "winogrande": ("acc,none", "WinoGrande accuracy", True),
    "arc_easy": ("acc_norm,none", "ARC-Easy normalized accuracy", True),
    "arc_challenge": ("acc_norm,none", "ARC-Challenge normalized accuracy", True),
    "mmlu": ("acc,none", "MMLU accuracy", True),
}


def task_value(label: str, task: str, metric: str) -> float:
    path = EXP / "task_quality/results" / label / task / "full/result.json"
    return float(json.loads(path.read_text())["metrics"][metric])


def main() -> None:
    out = EXP / "task_quality/report"
    out.mkdir(parents=True, exist_ok=True)
    base_nll = float(json.loads((EXP / "nll/raw/p00.json").read_text())["avg_nll"])
    uniform_nll = {row["policy_id"]: float(row["target_delta_nll"])
                   for row in csv.DictReader((EXP / "nll/prefill_only.csv").open())}
    anchors = {row["policy_id"]: float(row["e2e_median_ms"])
               for row in csv.DictReader((EXP / "speed/calibration/calibration.csv").open())}
    closure = {row["policy_id"]: row
               for row in csv.DictReader((EXP / "pareto/closure_summary.csv").open())}
    baseline_ms = float(closure["point_000"]["measured_e2e_ms"])
    ours_labels = (
        "point_003", "point_005", "point_007",
        "bridge_dense_nvfp4_072", "bridge_dense_nvfp4_088",
        "bridge_dense_nvfp4_104", "bridge_dense_nvfp4_120",
        "point_009", "point_011", "point_014",
    )
    rows: list[dict[str, object]] = []
    for label, name in UNIFORM.items():
        ms = baseline_ms if label == "p00" else anchors[label]
        row: dict[str, object] = {
            "family": "uniform", "policy": label, "display_name": name,
            "speed_ms": ms, "speedup_vs_dense_bf16": baseline_ms / ms,
            "delta_nll": uniform_nll[label],
        }
        for task, (metric, column, _) in TASKS.items():
            row[column] = task_value(label, task, metric)
        rows.append(row)
    for label in ours_labels:
        source = closure[label]
        ms = float(source["measured_e2e_ms"])
        row = {
            "family": "ours", "policy": label,
            "display_name": label.replace("point_", "p").replace("bridge_dense_nvfp4_", "bridge-"),
            "speed_ms": ms, "speedup_vs_dense_bf16": baseline_ms / ms,
            "delta_nll": float(source["measured_delta_nll"]),
        }
        for task, (metric, column, _) in TASKS.items():
            row[column] = task_value(label, task, metric)
        rows.append(row)
    fields = ["family", "policy", "display_name", "speed_ms", "speedup_vs_dense_bf16", "delta_nll", *(item[1] for item in TASKS.values())]
    with (out / "all_policy_task_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    def plot(metric: str, ylabel: str, suffix: str, higher_is_better: bool) -> None:
        ours = sorted((row for row in rows if row["family"] == "ours"), key=lambda row: float(row["speedup_vs_dense_bf16"]))
        uniform = [row for row in rows if row["family"] == "uniform"]
        plt.figure(figsize=(8.6, 5.4))
        plt.plot([row["speedup_vs_dense_bf16"] for row in ours], [row[metric] for row in ours],
                 "o-", color="#1f2937", linewidth=2.5, markersize=7.5, label="Ours (mixed policies)")
        plt.scatter([row["speedup_vs_dense_bf16"] for row in uniform], [row[metric] for row in uniform],
                    marker="s", s=90, color="#dc2626", label="Uniform baselines", zorder=3)
        for row in ours:
            plt.annotate(str(row["display_name"]), (float(row["speedup_vs_dense_bf16"]), float(row[metric])),
                         xytext=(3, 5), textcoords="offset points", fontsize=7.5)
        for row in uniform:
            plt.annotate(str(row["display_name"]), (float(row["speedup_vs_dense_bf16"]), float(row[metric])),
                         xytext=(4, -13), textcoords="offset points", fontsize=7.5, color="#991b1b")
        plt.xlabel("Measured E2E prefill speedup vs dense BF16")
        plt.ylabel(ylabel)
        direction = "higher is better" if higher_is_better else "lower is better"
        plt.title(f"Llama3.1-8B-Instruct prefill-only: speed vs {ylabel} ({direction})")
        plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(out / f"pareto_speed_vs_{suffix}.png", dpi=240); plt.close()

    plot("delta_nll", "Measured real-vLLM ΔNLL", "real_nll", False)
    for task, (_, column, higher) in TASKS.items():
        plot(column, column, task, higher)

    lines = [
        "# Llama3.1-8B-Instruct canonical prefill-only results", "",
        "All methods use the same `phase_hetero_mytest` vLLM runtime. Speeds are independently measured E2E prefill medians (five runs, B=8, input=2048). NLL and downstream task values are real phase-vLLM measurements; task evaluation uses conservative runtime allocation solely for stability and is not a timing measurement.", "",
        "## All measured policies", "",
        "| family | policy | speed (ms) | speedup | ΔNLL | WikiText PPL | WinoGrande | ARC-Easy | ARC-Challenge | MMLU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {family} | {display_name} | {speed_ms:.2f} | {speedup_vs_dense_bf16:.3f} | {delta_nll:.4f} | {wikitext:.4f} | {winogrande:.4f} | {arc_easy:.4f} | {arc_challenge:.4f} | {mmlu:.4f} |".format(
            **row, wikitext=row["WikiText word PPL"], winogrande=row["WinoGrande accuracy"],
            arc_easy=row["ARC-Easy normalized accuracy"], arc_challenge=row["ARC-Challenge normalized accuracy"], mmlu=row["MMLU accuracy"]))
    lines += ["", "## Artifacts", "", "- `all_policy_task_results.csv`: machine-readable result table.", "- `pareto_speed_vs_real_nll.png` and `pareto_speed_vs_{wikitext,winogrande,arc_easy,arc_challenge,mmlu}.png`: paper-facing Pareto plots."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
