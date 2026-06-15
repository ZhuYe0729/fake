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
    parser = argparse.ArgumentParser(description="Analyze sparse NVFP4 structural scenario loss deltas.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_root / "structural_scenarios"
    policies = read_csv(out / "sparse_nvfp4_structural_scenario_policies.csv")
    pairs = read_csv(out / "sparse_nvfp4_structural_scenario_pairs.csv")
    losses = read_csv(args.output_root / "loss" / "loss_samples_sparse_nvfp4_structural.csv")
    policy_by_id = {row["policy_id"]: row for row in policies}
    loss_by_id = {row["policy_id"]: row for row in losses}

    result_rows = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        low_id = f"{pair_id}_low_structural"
        high_id = f"{pair_id}_high_structural"
        low_loss = f(loss_by_id[low_id], "loss_delta_vs_dense")
        high_loss = f(loss_by_id[high_id], "loss_delta_vs_dense")
        measured_delta = high_loss - low_loss
        result_rows.append(
            {
                **pair,
                "low_policy_id": low_id,
                "high_policy_id": high_id,
                "low_loss_delta": low_loss,
                "high_loss_delta": high_loss,
                "measured_loss_delta": measured_delta,
                "direction_correct": int(measured_delta > 0),
                "low_composition": policy_by_id[low_id]["composition"],
                "high_composition": policy_by_id[high_id]["composition"],
            }
        )

    write_csv(out / "sparse_nvfp4_structural_scenario_results.csv", result_rows)
    plot_results(result_rows, out / "sparse_nvfp4_structural_scenario_loss_delta.png")
    write_report(out / "sparse_nvfp4_structural_scenario_loss_summary.md", result_rows)
    print(f"wrote {out / 'sparse_nvfp4_structural_scenario_loss_summary.md'}")


def plot_results(rows: list[dict[str, Any]], path: Path) -> None:
    xs = [f(row, "structural_delta") for row in rows]
    ys = [f(row, "measured_loss_delta") for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1, alpha=0.35)
    ax.set_xlabel("Structural proxy delta")
    ax.set_ylabel("Measured loss delta")
    ax.set_title("Sparse NVFP4 structural scenarios")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    measured = [f(row, "measured_loss_delta") for row in rows]
    structural = [f(row, "structural_delta") for row in rows]
    raw_gap = [f(row, "raw_rel_gap") for row in rows]
    lines = [
        "# Sparse NVFP4 Structural Scenario Loss Summary",
        "",
        f"Pairs: `{len(rows)}`",
        f"Direction accuracy: `{mean(f(row, 'direction_correct') for row in rows):.4f}`",
        f"Structural delta vs measured loss Pearson: `{pearson(structural, measured):.4f}`",
        f"Measured delta mean: `{mean(measured):.6f}`",
        f"Raw relative gap mean/max: `{mean(raw_gap):.6f}` / `{max(raw_gap):.6f}`",
        "",
        "| pair | raw rel gap | structural delta | measured loss delta | low loss | high loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_id']} | {f(row, 'raw_rel_gap'):.6f} | {f(row, 'structural_delta'):.6f} | "
            f"{f(row, 'measured_loss_delta'):.6f} | {f(row, 'low_loss_delta'):.6f} | {f(row, 'high_loss_delta'):.6f} |"
        )
    lines.extend(["", "## Plot", "", f"- `{path.parent / 'sparse_nvfp4_structural_scenario_loss_delta.png'}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / den_x / den_y if den_x and den_y else math.nan


if __name__ == "__main__":
    main()
