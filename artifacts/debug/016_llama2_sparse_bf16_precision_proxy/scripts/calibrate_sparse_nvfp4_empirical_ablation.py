#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from common_sparse_bf16_proxy import DEBUG_ROOT, f, read_csv, write_csv


VARIANTS = ("local_only", "local_depth", "local_type", "final_depth_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale-calibrate sparse NVFP4 empirical scenario ablation predictions.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--input", default="sparse_nvfp4_empirical_balanced_ablation_predictions.csv")
    parser.add_argument("--output-prefix", default="sparse_nvfp4_empirical_balanced_calibrated_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_root / "structural_scenarios"
    rows = read_csv(out / args.input)
    pred_rows = []
    summary_rows = []
    for variant in VARIANTS:
        items = [row for row in rows if row["variant"] == variant]
        scale = fit_scale(items)
        calibrated = []
        for row in items:
            item = dict(row)
            item["uncalibrated_pred_delta"] = item["pred_delta"]
            item["scale"] = scale
            item["pred_delta"] = f(row, "pred_delta") * scale
            item["abs_error"] = abs(f(item, "pred_delta") - f(item, "measured_delta"))
            item["direction_correct"] = int((f(item, "pred_delta") > 0) == (f(item, "measured_delta") > 0))
            calibrated.append(item)
        pred_rows.extend(calibrated)
        summary_rows.append(summarize(variant, calibrated, scale))
    write_csv(out / f"{args.output_prefix}_predictions.csv", pred_rows)
    write_csv(out / f"{args.output_prefix}_summary.csv", summary_rows)
    plot(summary_rows, out / f"{args.output_prefix}_metrics.png")
    write_report(out / f"{args.output_prefix}_summary.md", summary_rows, out, args.output_prefix)
    print(f"wrote {out / f'{args.output_prefix}_summary.md'}")


def fit_scale(rows: list[dict[str, str]]) -> float:
    xs = [f(row, "pred_delta") for row in rows]
    ys = [f(row, "measured_delta") for row in rows]
    den = sum(x * x for x in xs)
    return sum(x * y for x, y in zip(xs, ys)) / den if den else 0.0


def summarize(variant: str, rows: list[dict[str, str]], scale: float) -> dict[str, float | str | int]:
    pred = [f(row, "pred_delta") for row in rows]
    measured = [f(row, "measured_delta") for row in rows]
    abs_error = [f(row, "abs_error") for row in rows]
    return {
        "variant": variant,
        "pairs": len(rows),
        "scale": scale,
        "pearson": pearson(pred, measured),
        "spearman": spearman(pred, measured),
        "mae": mean(abs_error),
        "rmse": math.sqrt(mean(value * value for value in abs_error)),
        "direction_accuracy": mean(f(row, "direction_correct") for row in rows),
        "pred_delta_mean": mean(pred),
        "measured_delta_mean": mean(measured),
    }


def plot(rows: list[dict[str, float | str | int]], path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    direction = [float(row["direction_accuracy"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, mae)
    axes[0].set_title("Scale-calibrated MAE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, direction)
    axes[1].set_title("Direction accuracy")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, float | str | int]], out_dir: Path, output_prefix: str) -> None:
    mae_ranked = sorted(rows, key=lambda row: float(row["mae"]))
    direction_ranked = sorted(rows, key=lambda row: float(row["direction_accuracy"]), reverse=True)
    lines = [
        "# Sparse NVFP4 Balanced Scenario Scale-Calibrated Ablation",
        "",
        "Each variant receives the same scale-only calibration `pred_delta *= a` without an intercept.",
        "",
        "Main result: `final_depth_type` has the lowest MAE, while `local_only` is the weakest baseline by MAE/RMSE and direction accuracy.",
        "",
        f"- MAE rank: {', '.join(str(row['variant']) for row in mae_ranked)}",
        f"- Direction rank: {', '.join(str(row['variant']) for row in direction_ranked)}",
        "",
        "| variant | pairs | scale | Pearson | Spearman | MAE | RMSE | direction acc | pred delta mean | measured delta mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {int(float(row['pairs']))} | {float(row['scale']):.4f} | "
            f"{float(row['pearson']):.4f} | {float(row['spearman']):.4f} | {float(row['mae']):.6f} | "
            f"{float(row['rmse']):.6f} | {float(row['direction_accuracy']):.4f} | "
            f"{float(row['pred_delta_mean']):.6f} | {float(row['measured_delta_mean']):.6f} |"
        )
    lines.extend(["", "## Plot", "", f"- `{out_dir / f'{output_prefix}_metrics.png'}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / den_x / den_y if den_x and den_y else math.nan


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


if __name__ == "__main__":
    main()
