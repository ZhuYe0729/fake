#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from common_sparse_bf16_proxy import DEBUG_ROOT, f, read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sparse NVFP4 empirical structural scenario loss deltas.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--tag", default="empirical_structural")
    parser.add_argument("--scenario-prefix", default="sparse_nvfp4_empirical_scenario")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_root / "structural_scenarios"
    pairs = read_csv(out / f"{args.scenario_prefix}_pairs.csv")
    losses = read_csv(args.output_root / "loss" / f"loss_samples_sparse_nvfp4_{args.tag}.csv")
    loss_by_id = {row["policy_id"]: row for row in losses}
    rows = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        low_id = f"{pair_id}_low_empirical"
        high_id = f"{pair_id}_high_empirical"
        low_loss = f(loss_by_id[low_id], "loss_delta_vs_dense")
        high_loss = f(loss_by_id[high_id], "loss_delta_vs_dense")
        rows.append(
            {
                **pair,
                "low_policy_id": low_id,
                "high_policy_id": high_id,
                "low_loss_delta": low_loss,
                "high_loss_delta": high_loss,
                "measured_loss_delta": high_loss - low_loss,
                "direction_correct": int(high_loss > low_loss),
            }
        )
    write_csv(out / f"{args.scenario_prefix}_results.csv", rows)
    plot(rows, out / f"{args.scenario_prefix}_loss_delta.png")
    write_report(out / f"{args.scenario_prefix}_loss_summary.md", rows)
    print(f"wrote {out / f'{args.scenario_prefix}_loss_summary.md'}")


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([f(row, "empirical_score_delta") for row in rows], [f(row, "measured_loss_delta") for row in rows], alpha=0.85)
    ax.axhline(0, color="black", linewidth=1, alpha=0.35)
    ax.set_xlabel("Empirical structural score delta")
    ax.set_ylabel("Measured loss delta")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    measured = [f(row, "measured_loss_delta") for row in rows]
    score = [f(row, "empirical_score_delta") for row in rows]
    raw_gap = [f(row, "raw_rel_gap") for row in rows]
    lines = [
        "# Sparse NVFP4 Empirical Structural Scenario Loss Summary",
        "",
        f"Pairs: `{len(rows)}`",
        f"Direction accuracy: `{mean(f(row, 'direction_correct') for row in rows):.4f}`",
        f"Empirical score vs measured loss Pearson: `{pearson(score, measured):.4f}`",
        f"Measured delta mean: `{mean(measured):.6f}`",
        f"Raw relative gap mean/max: `{mean(raw_gap):.6f}` / `{max(raw_gap):.6f}`",
        "",
        "| pair | raw rel gap | empirical score delta | measured loss delta | low loss | high loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_id']} | {f(row, 'raw_rel_gap'):.6f} | {f(row, 'empirical_score_delta'):.6f} | "
            f"{f(row, 'measured_loss_delta'):.6f} | {f(row, 'low_loss_delta'):.6f} | {f(row, 'high_loss_delta'):.6f} |"
        )
    lines.extend(["", "## Plot", "", f"- `{path.with_name(path.stem.replace('_summary', '_delta') + '.png')}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / den_x / den_y if den_x and den_y else math.nan


if __name__ == "__main__":
    main()
