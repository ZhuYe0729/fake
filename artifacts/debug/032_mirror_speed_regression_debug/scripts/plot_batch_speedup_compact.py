#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = (
    "dense_default_amp",
    "uniform_dense_nvfp4",
    "uniform_sparse_bf16",
    "uniform_sparse_nvfp4",
    "ours_best",
)
BATCHES = (8, 16, 32)
LABELS = {
    "dense_default_amp": "Uncompressed AMP",
    "uniform_dense_nvfp4": "Dense NVFP4",
    "uniform_sparse_bf16": "Sparse BF16",
    "uniform_sparse_nvfp4": "Sparse NVFP4",
    "ours_best": "Ours",
}
COLORS = {
    "dense_default_amp": "#6b7280",
    "uniform_dense_nvfp4": "#4b5563",
    "uniform_sparse_bf16": "#d1d5db",
    "uniform_sparse_nvfp4": "#374151",
    "ours_best": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot compact MIRROR batch speedup bars.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    summary = build_summary(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output.with_suffix(".csv"), summary)
    plot(summary, args.output)
    print(f"wrote {args.output}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value == "" else float(value)


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_batch: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        batch = int(float(row["batch_size"]))
        if batch in BATCHES:
            by_batch.setdefault(batch, []).append(row)

    out: list[dict[str, Any]] = []
    for batch in BATCHES:
        group = by_batch[batch]
        baseline = next(row for row in group if row["label"] == "dense_default_amp")
        baseline_ms = f(baseline, "forward_mean_ms")
        by_label = {row["label"]: row for row in group}
        ours = min((row for row in group if row["group"] == "ours_candidate"), key=lambda row: f(row, "forward_mean_ms"))
        for label in ORDER:
            source = ours if label == "ours_best" else by_label[label]
            mean_ms = f(source, "forward_mean_ms")
            out.append(
                {
                    "batch_size": batch,
                    "label": label,
                    "display_label": LABELS[label],
                    "source_label": source["label"],
                    "forward_mean_ms": f"{mean_ms:.6f}",
                    "speedup_vs_uncompressed_amp": f"{baseline_ms / mean_ms:.6f}",
                }
            )
    return out


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    batches = list(BATCHES)
    width = 0.14
    x = list(range(len(batches)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, label in enumerate(ORDER):
        vals = [
            f(next(row for row in rows if int(row["batch_size"]) == batch and row["label"] == label), "speedup_vs_uncompressed_amp")
            for batch in batches
        ]
        offset = (i - (len(ORDER) - 1) / 2) * width
        bars = ax.bar([pos + offset for pos in x], vals, width=width, color=COLORS[label], label=LABELS[label])
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(batch) for batch in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup vs uncompressed AMP")
    ax.set_ylim(0, max(f(row, "speedup_vs_uncompressed_amp") for row in rows) * 1.24)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
