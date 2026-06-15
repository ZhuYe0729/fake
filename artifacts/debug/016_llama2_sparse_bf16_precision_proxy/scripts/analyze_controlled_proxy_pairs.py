#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from common_sparse_bf16_proxy import DEBUG_ROOT, f, read_csv, write_csv


METHODS = ("sparse_bf16", "dense_nvfp4", "sparse_nvfp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze controlled raw-local-matched proxy pair tests.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tag", default="controlled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    all_pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        pair_rows = analyze_method(args.output_root, method, args.tag)
        all_pair_rows.extend(pair_rows)
        summary_rows.extend(summary_for_method(method, pair_rows))
    out = args.output_root / "controlled"
    write_csv(out / "controlled_pair_results.csv", all_pair_rows)
    write_csv(out / "controlled_pair_summary.csv", summary_rows)
    plot_pair_deltas(all_pair_rows, out / "controlled_pair_delta_scatter.png")
    write_report(out / "controlled_pair_summary.md", summary_rows)
    print(f"wrote {out / 'controlled_pair_summary.md'}")


def analyze_method(output_root: Path, method: str, tag: str) -> list[dict[str, Any]]:
    policy_rows = read_csv(output_root / "controlled" / "policies" / f"controlled_policies_{method}.csv")
    loss_rows = read_csv(output_root / "loss" / f"loss_samples_{method}_{tag}.csv")
    loss_by_policy = {row["policy_id"]: row for row in loss_rows}
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in policy_rows:
        pairs.setdefault(row["pair_id"], {})[row["arm"]] = row
    out = []
    for pair_id, arms in sorted(pairs.items()):
        low = arms["low_final"]
        high = arms["high_final"]
        low_loss = loss_by_policy[low["policy_id"]]
        high_loss = loss_by_policy[high["policy_id"]]
        measured_delta = f(high_loss, "loss_delta_vs_dense") - f(low_loss, "loss_delta_vs_dense")
        final_delta = f(high, "final_proxy_sum") - f(low, "final_proxy_sum")
        raw_delta = f(high, "raw_error_sum") - f(low, "raw_error_sum")
        out.append(
            {
                "method": method,
                "pair_id": pair_id,
                "selected_modules": low["selected_modules"],
                "low_policy_id": low["policy_id"],
                "high_policy_id": high["policy_id"],
                "low_loss_delta": f(low_loss, "loss_delta_vs_dense"),
                "high_loss_delta": f(high_loss, "loss_delta_vs_dense"),
                "measured_pair_delta": measured_delta,
                "raw_pair_delta": raw_delta,
                "final_pair_delta": final_delta,
                "raw_abs_delta": abs(raw_delta),
                "final_abs_delta": abs(final_delta),
                "final_direction_correct": int(measured_delta > 0),
            }
        )
    return out


def summary_for_method(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [f(row, "measured_pair_delta") for row in rows]
    final = [f(row, "final_pair_delta") for row in rows]
    raw = [f(row, "raw_pair_delta") for row in rows]
    return [
        {
            "method": method,
            "pairs": len(rows),
            "final_direction_accuracy": mean(f(row, "final_direction_correct") for row in rows),
            "final_pair_pearson": pearson(final, measured),
            "raw_pair_pearson": pearson(raw, measured),
            "measured_delta_mean": mean(measured),
            "measured_delta_positive": sum(1 for value in measured if value > 0),
            "raw_abs_delta_mean": mean(abs(value) for value in raw),
            "final_abs_delta_mean": mean(abs(value) for value in final),
        }
    ]


def plot_pair_deltas(rows: list[dict[str, Any]], path: Path) -> None:
    methods = sorted({row["method"] for row in rows})
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.5), squeeze=False)
    for idx, method in enumerate(methods):
        ax = axes[0][idx]
        items = [row for row in rows if row["method"] == method]
        xs = [f(row, "final_pair_delta") for row in items]
        ys = [f(row, "measured_pair_delta") for row in items]
        cs = [f(row, "selected_modules") for row in items]
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", alpha=0.85)
        ax.axhline(0, color="black", linewidth=1, alpha=0.4)
        ax.axvline(0, color="black", linewidth=1, alpha=0.4)
        ax.set_title(method)
        ax.set_xlabel("Final proxy pair delta")
        ax.set_ylabel("Measured loss pair delta")
        ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="selected_modules")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Controlled Proxy Pair Summary",
        "",
        "| method | pairs | direction acc | final Pearson | raw Pearson | measured delta mean | raw abs delta mean | final abs delta mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {int(f(row, 'pairs'))} | {f(row, 'final_direction_accuracy'):.4f} | "
            f"{f(row, 'final_pair_pearson'):.4f} | {f(row, 'raw_pair_pearson'):.4f} | "
            f"{f(row, 'measured_delta_mean'):.6f} | {f(row, 'raw_abs_delta_mean'):.6f} | {f(row, 'final_abs_delta_mean'):.6f} |"
        )
    lines.extend(["", "## Plot", "", f"- `{path.parent / 'controlled_pair_delta_scatter.png'}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    px, py = zip(*pairs)
    mx, my = mean(px), mean(py)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in px))
    den_y = math.sqrt(sum((y - my) ** 2 for y in py))
    return num / den_x / den_y if den_x and den_y else math.nan


if __name__ == "__main__":
    main()
