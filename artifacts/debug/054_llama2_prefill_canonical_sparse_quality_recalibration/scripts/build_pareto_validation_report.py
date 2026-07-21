#!/usr/bin/env python3
"""Build the canonical-sparse solved-Pareto NLL/speed table and plot."""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"))
BASELINES = ROOT / "artifacts/debug/045_runtime_quality_consolidation/report/prefill_only_corrected_runtime_quality.csv"
UNIFORM_NLL = {"dense_bf16": "p00", "dense_nvfp4": "p01", "sparse_bf16": "p02", "sparse_nvfp4": "p03", "marlin_nvfp4": "p04"}
TASK_METRICS = {"wikitext": ("word_perplexity,none", "wikitext_word_ppl"),
                "winogrande": ("acc,none", "winogrande_acc"),
                "arc_easy": ("acc,none", "arc_easy_acc"),
                "arc_challenge": ("acc_norm,none", "arc_challenge_acc_norm"),
                "mmlu": ("acc,none", "mmlu_acc")}


def median_ms(directory: Path) -> float | None:
    files = sorted((directory / "runs").glob("measured_*.json"))
    if len(files) != 5:
        return None
    return statistics.median(json.loads(path.read_text())["elapsed_ms"] for path in files)


def task_value(label: str, task: str, metric: str) -> float | None:
    path = EXP / "pareto/validation/tasks" / label / task / "full/result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["metrics"].get(metric)


def main() -> None:
    out = EXP / "pareto/paper"
    out.mkdir(parents=True, exist_ok=True)
    predicted = {int(row["point_index"]): row for row in csv.DictReader((EXP / "pareto/pareto_points.csv").open())}
    nll_dir = EXP / "pareto/validation/nll"
    p0_path = nll_dir / "point_000.json"
    if not p0_path.exists():
        raise FileNotFoundError(p0_path)
    reference_nll = json.loads(p0_path.read_text())["avg_nll"]
    rows: list[dict[str, object]] = []
    for point, source in sorted(predicted.items()):
        label = f"point_{point:03d}"
        nll_path = nll_dir / f"{label}.json"
        if not nll_path.exists():
            continue
        actual = json.loads(nll_path.read_text())["avg_nll"]
        speed = median_ms(EXP / "pareto/validation/speed" / label)
        item = {"family": "ours", "policy": label, "point_index": point,
                     "speed_ms": speed, "actual_nll": actual,
                     "actual_delta_nll": actual - reference_nll,
                     "predicted_delta_nll": float(source["predicted_delta_nll"]),
                     "predicted_speedup": float(source["raw_linear_speedup_vs_dense"])}
        item.update({name: task_value(label, task, metric) for task, (metric, name) in TASK_METRICS.items()})
        rows.append(item)
    baseline_ms = next(row["speed_ms"] for row in rows if row["policy"] == "point_000")
    for row in rows:
        row["measured_speedup"] = None if row["speed_ms"] is None else baseline_ms / row["speed_ms"]
        row["delta_nll_error"] = row["actual_delta_nll"] - row["predicted_delta_nll"]
    fields = ["family", "policy", "point_index", "speed_ms", "measured_speedup", "actual_nll", "actual_delta_nll", "predicted_delta_nll", "delta_nll_error", "predicted_speedup", *(name for _, name in TASK_METRICS.values())]
    with (out / "ours_measured_pareto.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    quality = {row["policy_id"]: row for row in csv.DictReader((EXP / "reports/quality/predictions.csv").open())}
    uniform: list[dict[str, object]] = [{"family": "uniform", "policy": "dense_bf16", "speed_ms": baseline_ms,
                                         "measured_speedup": 1.0, "actual_delta_nll": 0.0,
                                         **{name: task_value("uniform_dense_bf16", task, metric) for task, (metric, name) in TASK_METRICS.items()}}]
    for row in csv.DictReader(BASELINES.open()):
        if row["model"] != "llama2" or row["family"] != "uniform" or row["policy"] not in UNIFORM_NLL:
            continue
        policy = row["policy"]
        if policy == "dense_bf16":
            continue
        phase_label = f"uniform_{policy}"
        phase_nll = nll_dir / f"{phase_label}.json"
        phase_speed = median_ms(EXP / "pareto/validation/speed" / phase_label)
        nll_value = (json.loads(phase_nll.read_text())["avg_nll"] - reference_nll
                     if phase_nll.exists() else float(quality[UNIFORM_NLL[policy]]["actual_delta_nll"]))
        speed = phase_speed if phase_speed is not None else float(row["e2e_ms"])
        item = {"family": "uniform", "policy": policy, "speed_ms": speed,
                        "measured_speedup": baseline_ms / speed,
                        "actual_delta_nll": nll_value}
        item.update({name: task_value(phase_label, task, metric) for task, (metric, name) in TASK_METRICS.items()})
        uniform.append(item)
    with (out / "uniform_references.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "policy", "speed_ms", "measured_speedup", "actual_delta_nll", *(name for _, name in TASK_METRICS.values())]); writer.writeheader(); writer.writerows(uniform)

    complete = [row for row in rows if row["measured_speedup"] is not None]
    plt.figure(figsize=(8.0, 5.2))
    plt.plot([row["measured_speedup"] for row in complete], [row["actual_delta_nll"] for row in complete], "o-", color="#1f2937", linewidth=2.4, markersize=8, label="Ours (canonical mixed Pareto)")
    plt.scatter([row["measured_speedup"] for row in uniform], [row["actual_delta_nll"] for row in uniform], marker="s", s=95, color="#dc2626", label="Uniform references", zorder=3)
    for row in complete:
        plt.annotate(row["policy"].replace("point_", "p"), (row["measured_speedup"], row["actual_delta_nll"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
    for row in uniform:
        plt.annotate(row["policy"], (row["measured_speedup"], row["actual_delta_nll"]), xytext=(5, -13), textcoords="offset points", fontsize=8, color="#991b1b")
    plt.xlabel("Measured E2E prefill speedup vs phase dense BF16")
    plt.ylabel("Measured real-vLLM ΔNLL vs BF16")
    plt.title("Llama2-7B-chat prefill-only Pareto validation")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "pareto_speed_vs_real_nll.png", dpi=240); plt.close()

    for task, (_, name) in TASK_METRICS.items():
        ours_task = [row for row in complete if row[name] is not None]
        uniform_task = [row for row in uniform if row[name] is not None]
        if not ours_task or not uniform_task:
            continue
        plt.figure(figsize=(8.0, 5.2))
        plt.plot([row["measured_speedup"] for row in ours_task], [row[name] for row in ours_task], "o-", color="#1f2937", linewidth=2.4, markersize=8, label="Ours (canonical mixed Pareto)")
        plt.scatter([row["measured_speedup"] for row in uniform_task], [row[name] for row in uniform_task], marker="s", s=95, color="#dc2626", label="Uniform references", zorder=3)
        for row in ours_task:
            plt.annotate(row["policy"].replace("point_", "p"), (row["measured_speedup"], row[name]), xytext=(4, 5), textcoords="offset points", fontsize=8)
        for row in uniform_task:
            plt.annotate(row["policy"], (row["measured_speedup"], row[name]), xytext=(5, -13), textcoords="offset points", fontsize=8, color="#991b1b")
        label = "WikiText word perplexity (lower is better)" if task == "wikitext" else f"{task.replace('_', ' ').title()} accuracy"
        plt.xlabel("Measured E2E prefill speedup vs phase dense BF16")
        plt.ylabel(label); plt.title(f"Llama2-7B-chat prefill-only: speed vs {task.replace('_', ' ')}")
        plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / f"pareto_speed_vs_{task}.png", dpi=240); plt.close()

    lines = ["# Llama2-7B-chat canonical sparse prefill Pareto", "", "`ours` NLL is measured through real vLLM phase-heterogeneous inference on 100 fixed WikiText blocks. Each speed is the median of five loaded-vLLM prefill runs (batch 8, input 2048). Uniform compressed baselines are remeasured through the identical phase runtime; dense BF16 is the shared phase reference.", "", "## Solved mixed policies", "", "| policy | speed (ms) | speedup | real ΔNLL | predicted ΔNLL | residual |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        def show(value: object) -> str: return "—" if value is None else f"{float(value):.4f}"
        lines.append(f"| {row['policy']} | {show(row['speed_ms'])} | {show(row['measured_speedup'])} | {show(row['actual_delta_nll'])} | {show(row['predicted_delta_nll'])} | {show(row['delta_nll_error'])} |")
    lines += ["", "## Uniform references", "", "| policy | speed (ms) | speedup | real ΔNLL |", "|---|---:|---:|---:|"]
    for row in uniform:
        lines.append(f"| {row['policy']} | {row['speed_ms']:.4f} | {row['measured_speedup']:.4f} | {row['actual_delta_nll']:.4f} |")
    all_task_rows = [*uniform, *rows]
    lines += ["", "## Real-vLLM downstream tasks", "", "| family | policy | WikiText PPL | WinoGrande acc | ARC-Easy acc | ARC-Challenge norm acc | MMLU acc |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in all_task_rows:
        values = ["—" if row[name] is None else f"{float(row[name]):.4f}" for _, name in TASK_METRICS.values()]
        lines.append(f"| {row['family']} | {row['policy']} | " + " | ".join(values) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    merged_fields = ["family", "policy", "speed_ms", "measured_speedup", "actual_delta_nll", *(name for _, name in TASK_METRICS.values())]
    with (out / "all_methods_measured.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields); writer.writeheader()
        writer.writerows([{field: row.get(field) for field in merged_fields} for row in [*uniform, *rows]])


if __name__ == "__main__":
    main()
