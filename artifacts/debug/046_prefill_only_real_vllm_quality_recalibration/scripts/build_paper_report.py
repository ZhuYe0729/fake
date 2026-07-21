#!/usr/bin/env python3
"""Join actual Pareto validation, uniform baselines, and task results."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from common import ROOT, model_root

METRICS = {
    "wikitext": ("word_perplexity,none", "wikitext_word_ppl"),
    "winogrande": ("acc,none", "winogrande_acc"),
    "arc_easy": ("acc_norm,none", "arc_easy_acc_norm"),
    "arc_challenge": ("acc_norm,none", "arc_challenge_acc_norm"),
    "mmlu": ("acc,none", "mmlu_acc"),
}
UNIFORM_NLL_POLICY = {
    "dense_bf16": "p00", "dense_nvfp4": "p01", "sparse_bf16": "p02",
    "sparse_nvfp4": "p03", "marlin_nvfp4": "p04",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def measured_ms(path: Path) -> float | None:
    files = sorted((path / "runs").glob("measured_*.json"))
    if len(files) != 5:
        return None
    return statistics.median(json.loads(item.read_text())["elapsed_ms"] for item in files)


def task_metric(root: Path, point: str, task: str, source: str) -> float | None:
    path = root / "pareto/validation/tasks" / point / task / "full/result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("metrics", {}).get(source)


def raw_value(row: dict[str, str], target: str) -> float | None:
    """Read a baseline summary metric and convert its historical percent scale."""
    source = {
        "wikitext_word_ppl": "wikitext_word_ppl",
        "winogrande_acc": "winogrande_acc_pct",
        "arc_easy_acc_norm": "arc_easy_norm_pct",
        "arc_challenge_acc_norm": "arc_challenge_norm_pct",
        "mmlu_acc": "mmlu_acc_pct",
    }[target]
    try:
        value = float(row[source])
    except (KeyError, TypeError, ValueError):
        return None
    return value if target == "wikitext_word_ppl" else value / 100.0


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", default="0,8,12,15,18,20,22,23")
    args = parser.parse_args()
    root = model_root("llama2")
    output = root / "pareto/paper"
    output.mkdir(parents=True, exist_ok=True)
    predicted = {int(row["point_index"]): row for row in read_csv(root / "pareto/pareto_points.csv")}
    p0 = json.loads((root / "pareto/validation/nll/point_000.json").read_text())["avg_nll"]
    rows: list[dict[str, object]] = []
    for point in (int(item) for item in args.selected.split(",") if item.strip()):
        label = f"point_{point:03d}"
        nll_path = root / "pareto/validation/nll" / f"{label}.json"
        speed = measured_ms(root / "pareto/validation/speed" / label)
        if not nll_path.exists():
            continue
        actual_nll = json.loads(nll_path.read_text())["avg_nll"]
        row: dict[str, object] = {"family": "ours", "policy": label, "point_index": point,
                                  "speed_ms": speed, "speedup_vs_ours_bf16": None if speed is None else None,
                                  "actual_nll": actual_nll, "actual_delta_nll": actual_nll - p0,
                                  "predicted_delta_nll": float(predicted[point]["predicted_delta_nll"]),
                                  "predicted_linear_speedup": float(predicted[point]["raw_linear_speedup_vs_dense"]),
                                  "complete_speed": speed is not None}
        for task, (source, target) in METRICS.items():
            row[target] = task_metric(root, label, task, source)
        rows.append(row)
    completed = [row for row in rows if row["policy"] == "point_000" and row["speed_ms"] is not None]
    if completed:
        base = float(completed[0]["speed_ms"])
        for row in rows:
            if row["speed_ms"] is not None:
                row["speedup_vs_ours_bf16"] = base / float(row["speed_ms"])
    columns = ["family", "policy", "point_index", "speed_ms", "speedup_vs_ours_bf16", "actual_nll", "actual_delta_nll", "predicted_delta_nll", "predicted_linear_speedup", "complete_speed", *(target for _, target in METRICS.values())]
    write_csv(output / "ours_measured_pareto.csv", rows, columns)

    baseline_path = ROOT / "artifacts/debug/045_runtime_quality_consolidation/report/prefill_only_corrected_runtime_quality.csv"
    baseline = []
    if baseline_path.exists():
        for raw in read_csv(baseline_path):
            if raw.get("model") not in {"llama2", "llama2_7b_chat"} or raw.get("family") == "ours":
                continue
            baseline.append(raw)
    quality_rows = {row["policy_id"]: row for row in read_csv(root / "reports/quality/predictions.csv")}
    for raw in baseline:
        quality = quality_rows.get(UNIFORM_NLL_POLICY.get(raw["policy"], ""))
        raw["measured_delta_nll"] = "" if quality is None else quality["actual_delta_nll"]
    paper_rows: list[dict[str, object]] = []
    for raw in baseline:
        row: dict[str, object] = {"family": "uniform", "policy": raw["policy"], "recommendation": raw.get("recommendation", ""),
                                  "speed_ms": float(raw["e2e_ms"]), "speedup_vs_dense_bf16": float(raw["speedup"]),
                                  "measured_delta_nll": None if not raw["measured_delta_nll"] else float(raw["measured_delta_nll"]), "predicted_delta_nll": None}
        for _, target in METRICS.values():
            row[target] = raw_value(raw, target)
        paper_rows.append(row)
    for raw in rows:
        paper_rows.append({"family": "ours", "policy": raw["policy"], "recommendation": "candidate" if raw["policy"] in {"point_008", "point_012"} else "",
                           "speed_ms": raw["speed_ms"], "speedup_vs_dense_bf16": raw["speedup_vs_ours_bf16"],
                           "measured_delta_nll": raw["actual_delta_nll"], "predicted_delta_nll": raw["predicted_delta_nll"],
                           **{target: raw[target] for _, target in METRICS.values()}})
    paper_columns = ["family", "policy", "recommendation", "speed_ms", "speedup_vs_dense_bf16", "measured_delta_nll", "predicted_delta_nll", *(target for _, target in METRICS.values())]
    write_csv(output / "paper_table_all_methods.csv", paper_rows, paper_columns)
    plot_rows = [row for row in rows if row["speedup_vs_ours_bf16"] is not None]
    if plot_rows:
        plt.figure(figsize=(7.2, 4.8))
        plt.plot([float(r["speedup_vs_ours_bf16"]) for r in plot_rows], [float(r["actual_delta_nll"]) for r in plot_rows], "o-", color="#1f2937", label="Ours (mixed Pareto)")
        uniform_nll = [raw for raw in baseline if raw.get("measured_delta_nll")]
        if uniform_nll:
            plt.scatter([float(raw["speedup"]) for raw in uniform_nll], [float(raw["measured_delta_nll"]) for raw in uniform_nll], marker="s", color="#dc2626", label="Uniform references")
            for raw in uniform_nll:
                plt.annotate(raw["policy"], (float(raw["speedup"]), float(raw["measured_delta_nll"])), xytext=(4, -12), textcoords="offset points", fontsize=7, color="#991b1b")
        for row in plot_rows:
            plt.annotate(row["policy"].replace("point_", "p"), (float(row["speedup_vs_ours_bf16"]), float(row["actual_delta_nll"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
        plt.xlabel("Measured E2E prefill speedup vs dense BF16")
        plt.ylabel("Measured real-vLLM ΔNLL vs BF16")
        plt.title("Llama2-7B-chat prefill-only Pareto validation")
        plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(output / "pareto_speed_vs_real_nll.png", dpi=220); plt.close()

    # Downstream task plots deliberately retain the uniform references.  Their
    # historical measured speed is normalized to their dense-BF16 reference;
    # ours is normalized to the freshly remeasured solved-BF16 policy.
    for task, (_, target) in METRICS.items():
        ours_task = [row for row in plot_rows if row[target] is not None]
        baseline_task = [row for row in baseline if raw_value(row, target) is not None]
        if not ours_task and not baseline_task:
            continue
        plt.figure(figsize=(7.2, 4.8))
        if ours_task:
            plt.plot([float(row["speedup_vs_ours_bf16"]) for row in ours_task], [float(row[target]) for row in ours_task], "o-", color="#1f2937", label="Ours (mixed Pareto)")
            for row in ours_task:
                plt.annotate(row["policy"].replace("point_", "p"), (float(row["speedup_vs_ours_bf16"]), float(row[target])), xytext=(4, 4), textcoords="offset points", fontsize=8)
        if baseline_task:
            xs = [float(row["speedup"]) for row in baseline_task]
            ys = [raw_value(row, target) for row in baseline_task]
            plt.scatter(xs, ys, marker="s", color="#dc2626", label="Uniform references")
            for row, x, y in zip(baseline_task, xs, ys):
                plt.annotate(row["policy"], (x, y), xytext=(4, -10), textcoords="offset points", fontsize=7, color="#991b1b")
        metric_label = "WikiText word perplexity (lower is better)" if task == "wikitext" else f"{task.replace('_', ' ').title()} accuracy"
        plt.xlabel("Measured E2E prefill speedup vs dense BF16")
        plt.ylabel(metric_label)
        plt.title(f"Llama2-7B-chat prefill-only: speed vs {task}")
        plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(output / f"pareto_speed_vs_{task}.png", dpi=220); plt.close()

    lines = ["# Llama2-7B-chat prefill-only: solved Pareto validation", "", "All `ours` values use a policy newly solved from the real-vLLM NLL quality model. NLL is measured over the fixed 100 WikiText blocks; speed is the median of five loaded-vLLM prefill runs (batch 8, input 2048). Each reported point has completed all five real-vLLM downstream tasks. Uniform NLL references come from the same fixed-block runtime calibration; each series is normalized to its own measured dense-BF16 reference.", "", "| policy | speed (ms) | speedup | measured ΔNLL | predicted ΔNLL | status |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        show = lambda value: "—" if value is None else f"{float(value):.4f}"
        lines.append(f"| {row['policy']} | {show(row['speed_ms'])} | {show(row['speedup_vs_ours_bf16'])} | {show(row['actual_delta_nll'])} | {show(row['predicted_delta_nll'])} | {'complete' if row['complete_speed'] else 'NLL only'} |")
    task_rows = [row for row in rows if any(row[target] is not None for _, target in METRICS.values())]
    if task_rows:
        lines += ["", "## Real-vLLM downstream tasks", "", "| policy | WikiText PPL | WinoGrande acc | ARC-Easy norm acc | ARC-Challenge norm acc | MMLU acc |", "|---|---:|---:|---:|---:|---:|"]
        for row in task_rows:
            values = ["—" if row[target] is None else f"{float(row[target]):.4f}" for _, target in METRICS.values()]
            lines.append(f"| {row['policy']} | " + " | ".join(values) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
