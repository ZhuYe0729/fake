#!/usr/bin/env python
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_ROOT = REPO_ROOT / "artifacts/debug/020_fakevlm_uniform_accuracy/outputs"
OUTPUT_ROOT = Path(__file__).resolve().parent

METHODS = (
    "dense_bf16",
    "sparse_bf16",
    "dense_nvfp4",
    "sparse_nvfp4",
    "marlin_weight_only",
    "dense_nvfp4_prefill_marlin_decode",
)

SAMPLE_SIZES = (25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000)
NUM_SEEDS = 30
ERROR_THRESHOLDS = (0.005, 0.01, 0.02)


def load_predictions(method: str) -> list[dict]:
    path = PREDICTIONS_ROOT / method / "predictions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def accuracy(predictions: list[dict]) -> float:
    correct = sum(1 for p in predictions if p["label"] == p["pred"])
    return correct / len(predictions)


def run_analysis(method: str, predictions: list[dict]) -> dict:
    full_acc = accuracy(predictions)
    total = len(predictions)
    raw_rows = []
    stats_rows = []

    for n in SAMPLE_SIZES:
        if n > total:
            continue
        accs = []
        for seed in range(NUM_SEEDS):
            rng = random.Random(seed)
            subset = rng.sample(predictions, n)
            acc = accuracy(subset)
            abs_err = abs(acc - full_acc)
            accs.append(acc)
            raw_rows.append((n, seed, acc, abs_err))

        arr = np.array(accs)
        abs_errs = np.abs(arr - full_acc)
        stats_rows.append(
            (
                n,
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.mean(abs_errs)),
                float(np.max(abs_errs)),
            )
        )

    return {
        "full_acc": full_acc,
        "total": total,
        "raw_rows": raw_rows,
        "stats_rows": stats_rows,
    }


def write_csv(path: Path, headers: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def threshold_report(stats_rows: list[tuple], full_acc: float) -> dict:
    report = {}
    for threshold in ERROR_THRESHOLDS:
        for n, mean_acc, std_acc, min_acc, max_acc, mean_abs_err, max_abs_err in stats_rows:
            if max_abs_err <= threshold:
                report[f"max_err_le_{threshold:.3f}"] = (n, max_abs_err)
                break
        else:
            report[f"max_err_le_{threshold:.3f}"] = None
        for n, mean_acc, std_acc, min_acc, max_acc, mean_abs_err, max_abs_err in stats_rows:
            if mean_abs_err <= threshold:
                report[f"mean_err_le_{threshold:.3f}"] = (n, mean_abs_err)
                break
        else:
            report[f"mean_err_le_{threshold:.3f}"] = None
    return report


def plot_method(method: str, stats_rows: list[tuple], full_acc: float, out_dir: Path) -> None:
    ns = [r[0] for r in stats_rows]
    means = [r[1] for r in stats_rows]
    stds = [r[2] for r in stats_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(y=full_acc, color="gray", linestyle="--", linewidth=0.8, label=f"Full accuracy ({full_acc:.4f})")

    ax.fill_between(ns, np.array(means) - 2 * np.array(stds), np.array(means) + 2 * np.array(stds),
                    alpha=0.12, color="steelblue", label=r"$\pm 2\sigma$")
    ax.fill_between(ns, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                    alpha=0.18, color="steelblue", label=r"$\pm 1\sigma$")
    ax.plot(ns, means, "o-", color="steelblue", markersize=4, linewidth=1.2, label="Mean accuracy")

    ax.set_xlabel("Sample size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{method} — accuracy vs sample size")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, max(ns) * 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_samples.png", dpi=150)
    plt.close(fig)


def plot_comparison(all_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    for method, color in zip(METHODS, colors):
        result = all_results[method]
        ns = [r[0] for r in result["stats_rows"]]
        means = [r[1] for r in result["stats_rows"]]
        ax.plot(ns, means, "o-", color=color, markersize=3, linewidth=1.0, label=f"{method} ({result['full_acc']:.4f})")

    ax.set_xlabel("Sample size")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("All methods — mean accuracy vs sample size")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_xlim(0, max(SAMPLE_SIZES) * 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "all_methods_comparison.png", dpi=150)
    plt.close(fig)


def plot_error(all_results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    for method, color in zip(METHODS, colors):
        result = all_results[method]
        ns = [r[0] for r in result["stats_rows"]]
        mean_errs = [r[5] for r in result["stats_rows"]]
        ax.plot(ns, mean_errs, "o-", color=color, markersize=3, linewidth=1.0, label=method)

    for threshold in ERROR_THRESHOLDS:
        ax.axhline(y=threshold, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.text(max(SAMPLE_SIZES) * 1.005, threshold, f"{threshold*100:.1f}%", fontsize=7, va="center", alpha=0.6)

    ax.set_xlabel("Sample size")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("All methods — mean absolute error vs sample size")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_xlim(0, max(SAMPLE_SIZES) * 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_samples.png", dpi=150)
    plt.close(fig)


def plot_error_percentage(all_results: dict, total: int, out_dir: Path) -> None:
    """Plot mean absolute error (in %) vs sample size as percentage of total."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    method_labels = {
        "dense_bf16": "dense BF16",
        "sparse_bf16": "sparse BF16",
        "dense_nvfp4": "dense NVFP4",
        "sparse_nvfp4": "sparse NVFP4",
        "marlin_weight_only": "marlin weight-only",
        "dense_nvfp4_prefill_marlin_decode": "dense NVFP4 prefill + marlin decode",
    }

    # Compute max y across the 5 high-accuracy methods to auto-zoom
    y_max = 0
    for method, color in zip(METHODS, colors):
        result = all_results[method]
        pcts = [r[0] / total * 100 for r in result["stats_rows"]]
        mean_err_pcts = [r[5] * 100 for r in result["stats_rows"]]
        label = method_labels.get(method, method)
        ax.plot(pcts, mean_err_pcts, "o-", color=color, markersize=4, linewidth=1.2, label=label)
        if method != "sparse_nvfp4":
            y_max = max(y_max, max(mean_err_pcts))

    for threshold in ERROR_THRESHOLDS:
        t_pct = threshold * 100
        if t_pct > y_max * 1.5:
            break
        ax.axhline(y=t_pct, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.text(0.5, t_pct + 0.02, f"{t_pct:.1f}%", fontsize=7, va="bottom", alpha=0.5, color="gray")

    ax.set_xlabel("Sample size (% of full dataset)")
    ax.set_ylabel("Mean absolute error (%)")
    ax.set_title("Accuracy estimation error vs sample size")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_xlim(0, max(SAMPLE_SIZES) / total * 100 * 1.02)
    ax.set_ylim(0, y_max * 1.15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_percentage.png", dpi=150)
    plt.close(fig)


def write_markdown_report(all_results: dict, out_dir: Path) -> None:
    lines = [
        "# FakeVLM Sample Size Accuracy Analysis",
        "",
        "## Overview",
        "",
        "This analysis determines how many test samples are needed to reliably estimate the true accuracy of FakeVLM models. "
        "It uses statistical subsampling of the full 5000-sample predictions from `020_fakevlm_uniform_accuracy`, "
        f"with {NUM_SEEDS} random seeds per sample size.",
        "",
        "## Full Accuracy (5000 samples)",
        "",
        "| Method | Accuracy | Correct | Wrong |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        r = all_results[method]
        lines.append(f"| `{method}` | {r['full_acc']:.6f} | {int(r['full_acc'] * r['total'])} | {r['total'] - int(r['full_acc'] * r['total'])} |")

    lines += [
        "",
        "## Recommended Sample Sizes",
        "",
        "The table below shows the minimum sample size N where the error metric drops below the threshold. "
        '"Max error" is the worst-case across all 30 seeds; "mean error" is the average.',
        "",
        "| Method | Full Acc | N for max err ≤0.5% | N for max err ≤1% | N for max err ≤2% | N for mean err ≤0.5% | N for mean err ≤1% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for method in METHODS:
        r = all_results[method]
        tr = r["thresholds"]
        def fmt(val):
            if val is None:
                return ">4000"
            return str(val[0])
        lines.append(
            f"| `{method}` | {r['full_acc']:.4f} | {fmt(tr.get('max_err_le_0.005'))} | {fmt(tr.get('max_err_le_0.010'))} | "
            f"{fmt(tr.get('max_err_le_0.020'))} | {fmt(tr.get('mean_err_le_0.005'))} | {fmt(tr.get('mean_err_le_0.010'))} |"
        )

    lines += [
        "",
        "## Detailed Stats (dense_bf16)",
        "",
        "Full table for the primary baseline method:",
        "",
        "| N | Mean Acc | Std | Min | Max | Mean Abs Err | Max Abs Err |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_results["dense_bf16"]["stats_rows"]:
        lines.append(
            f"| {row[0]} | {row[1]:.6f} | {row[2]:.6f} | {row[3]:.6f} | {row[4]:.6f} | {row[5]:.6f} | {row[6]:.6f} |"
        )

    lines += [
        "",
        "## Plots",
        "",
        "- `outputs/<method>/accuracy_vs_samples.png` — per-method accuracy convergence with ±1σ/±2σ bands",
        "- `summary/all_methods_comparison.png` — all methods overlaid",
        "- `summary/error_vs_samples.png` — mean absolute error vs N for all methods",
        "- `summary/error_vs_percentage.png` — mean absolute error (%) vs sample size as % of full dataset",
        "",
        "## Interpretation",
        "",
        "The accuracy converges quickly because the model is highly accurate (~98.6% for most methods). "
        "With only ~100 samples, the estimate is already within ±2-3% of the true accuracy most of the time. "
        "By ~500 samples, the error is typically under ±1%.",
        "",
        "For the outlier `sparse_nvfp4` method (~76.9% accuracy), convergence is even faster in absolute terms "
        "because the variance of a binomial proportion is maximal at p=0.5 and decreases as p approaches 0 or 1.",
        "",
        "**Recommendation**: 500-1000 samples provides a good balance between speed and accuracy (error < ±1%). "
        "For quick smoke tests, 200 samples gives a reasonable estimate (error < ±2%).",
    ]

    report_path = out_dir / "sample_size_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_results = {}
    for method in METHODS:
        print(f"[{method}] loading predictions...")
        predictions = load_predictions(method)
        print(f"[{method}] {len(predictions)} samples, full accuracy = {accuracy(predictions):.6f}")
        result = run_analysis(method, predictions)
        result["thresholds"] = threshold_report(result["stats_rows"], result["full_acc"])
        all_results[method] = result

        out_dir = OUTPUT_ROOT / "outputs" / method
        write_csv(
            out_dir / "sample_size_accuracy.csv",
            ["n", "seed", "accuracy", "abs_error"],
            result["raw_rows"],
        )
        write_csv(
            out_dir / "sample_size_stats.csv",
            ["n", "mean_acc", "std_acc", "min_acc", "max_acc", "mean_abs_err", "max_abs_err"],
            result["stats_rows"],
        )
        plot_method(method, result["stats_rows"], result["full_acc"], out_dir)
        print(f"[{method}] done")

    print("[summary] generating cross-method plots and report...")
    plot_comparison(all_results, OUTPUT_ROOT / "summary")
    plot_error(all_results, OUTPUT_ROOT / "summary")
    plot_error_percentage(all_results, 5000, OUTPUT_ROOT / "summary")

    summary_rows = []
    for method in METHODS:
        r = all_results[method]
        tr = r["thresholds"]
        row = [method, r["full_acc"]]
        for t in ERROR_THRESHOLDS:
            v = tr.get(f"max_err_le_{t:.3f}")
            row.append(v[0] if v else ">4000")
        for t in ERROR_THRESHOLDS:
            v = tr.get(f"mean_err_le_{t:.3f}")
            row.append(v[0] if v else ">4000")
        summary_rows.append(tuple(row))

    headers = ["method", "full_acc"]
    headers += [f"n_max_err_le_{t:.3f}" for t in ERROR_THRESHOLDS]
    headers += [f"n_mean_err_le_{t:.3f}" for t in ERROR_THRESHOLDS]
    write_csv(OUTPUT_ROOT / "summary" / "sample_size_summary.csv", headers, summary_rows)

    write_markdown_report(all_results, OUTPUT_ROOT / "summary")
    print("[done] all outputs written")


if __name__ == "__main__":
    main()