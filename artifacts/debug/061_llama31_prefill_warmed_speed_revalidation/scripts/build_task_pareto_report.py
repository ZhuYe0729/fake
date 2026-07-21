#!/usr/bin/env python3
"""Build paper-review tables and task Pareto plots from measured 061 closure."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from scenario import EXP, SOURCE

OURS = ("point_000", "point_003", "point_005", "point_007", "point_009", "point_011", "point_013", "point_014")
UNIFORM = {"dense_bf16": "p00", "dense_nvfp4": "p01", "sparse_bf16": "p02", "sparse_nvfp4": "p03", "w4a16_ours": "p04"}
TASKS = {
    "wikitext": ("word_perplexity,none", "WikiText word perplexity", False),
    "winogrande": ("acc,none", "Winogrande accuracy", True),
    "arc_easy": ("acc,none", "ARC-Easy accuracy", True),
    "arc_challenge": ("acc_norm,none", "ARC-Challenge normalized accuracy", True),
    "mmlu": ("acc,none", "MMLU accuracy", True),
}


def load_result(root: Path, policy: str, task: str, key: str) -> float:
    return float(json.loads((root / policy / task / "full/result.json").read_text())["metrics"][key])


def main() -> None:
    report = EXP / "report"; figures = report / "task_pareto"
    figures.mkdir(parents=True, exist_ok=True)
    speed_dir = EXP / "speed/calibration/runs"
    closure_speed = EXP / "pareto/closure/speed"
    ours_tasks = EXP / "task_quality/results"
    uniform_tasks = SOURCE / "task_quality/results"
    reference_ms = float(json.loads((closure_speed / "point_000.json").read_text())["median_ms"])
    rows: list[dict[str, object]] = []
    for label in OURS:
        ms = float(json.loads((closure_speed / f"{label}.json").read_text())["median_ms"])
        row: dict[str, object] = {"family": "ours", "policy": label, "speed_ms": ms, "speedup": reference_ms / ms}
        for task, (key, _, _) in TASKS.items(): row[task] = load_result(ours_tasks, label, task, key)
        rows.append(row)
    for display, label in UNIFORM.items():
        ms = float(json.loads((speed_dir / f"{label}.json").read_text())["median_ms"])
        row = {"family": "uniform", "policy": display, "speed_ms": ms, "speedup": reference_ms / ms}
        for task, (key, _, _) in TASKS.items(): row[task] = load_result(uniform_tasks, label, task, key)
        rows.append(row)
    with (report / "measured_task_pareto.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "policy", "speed_ms", "speedup", *TASKS])
        writer.writeheader(); writer.writerows(rows)
    for task, (_, title, higher_is_better) in TASKS.items():
        ours = [row for row in rows if row["family"] == "ours"]
        uniform = [row for row in rows if row["family"] == "uniform"]
        plt.figure(figsize=(8.0, 5.2))
        plt.plot([row["speedup"] for row in ours], [row[task] for row in ours], "o-", color="#1f2937", linewidth=2.4, markersize=7, label="Ours (mixed Pareto)")
        plt.scatter([row["speedup"] for row in uniform], [row[task] for row in uniform], marker="s", s=90, color="#dc2626", label="Uniform baselines", zorder=3)
        for row in ours:
            plt.annotate(str(row["policy"]).replace("point_", "p"), (row["speedup"], row[task]), xytext=(4, 5), textcoords="offset points", fontsize=8)
        for row in uniform:
            plt.annotate(str(row["policy"]), (row["speedup"], row[task]), xytext=(5, -13), textcoords="offset points", fontsize=8, color="#991b1b")
        plt.xlabel("Measured E2E prefill speedup vs dense BF16")
        plt.ylabel(title)
        plt.title(f"Llama-3.1-8B-Instruct prefill-only: speed vs {title}")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout(); plt.savefig(figures / f"speed_vs_{task}.png", dpi=200); plt.close()
    lines = ["# Llama-3.1-8B-Instruct prefill-only: measured task Pareto", "", "All speed values are 061 warmed phase-vLLM medians (B=8, L=2048, one warmup + five timed requests). `ours` task scores are newly measured with real phase-vLLM canonical checkpoints. Uniform task scores are the existing frozen 058 measurements; p00 is identical to 061 point_000.", "", "| family | policy | speed (ms) | speedup | WikiText PPL ↓ | Winogrande ↑ | ARC-Easy ↑ | ARC-Challenge ↑ | MMLU ↑ |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda item: (item["family"] != "uniform", float(item["speedup"]))):
        lines.append("| {family} | {policy} | {speed_ms:.2f} | {speedup:.2f}x | {wikitext:.4f} | {winogrande:.4f} | {arc_easy:.4f} | {arc_challenge:.4f} | {mmlu:.4f} |".format(**row))
    lines += ["", "## Suggested points", "", "- High-quality: `point_005` (1.44x; close to BF16 on all five tasks).", "- Balanced / speed-first: `point_009` (1.91x; faster than uniform dense-NVFP4 and vastly higher task quality than uniform sparse-NVFP4).", "- Max-speed: `point_014` (2.26x; report as an explicit quality-sacrificing endpoint)."]
    (report / "task_pareto_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
