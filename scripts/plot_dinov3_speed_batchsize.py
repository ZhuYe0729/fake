#!/usr/bin/env python3
"""Plot DINOv3 batch-size speed comparison from benchmark CSV files."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("artifacts/results")
OUTPUT_PATH = RESULTS_DIR / "dinov3_speed_batchsize.png"

SERIES = [
    (
        "Dense FP32",
        RESULTS_DIR / "dinov3_vit7b16_dense" / "speed.csv",
        "#3b5bdb",
        "o",
    ),
    (
        "CUTLASS dense NVFP4",
        RESULTS_DIR / "dinov3_vit7b16_cutlass_nvfp4" / "speed.csv",
        "#f08c00",
        "s",
    ),
    (
        "CUTLASS sparse NVFP4",
        RESULTS_DIR / "dinov3_vit7b16_cutlass_sparse_nvfp4" / "speed_storage.csv",
        "#0ca678",
        "^",
    ),
]


def read_latest_sweep(path: Path) -> list[dict[str, float | int | datetime]]:
    rows: list[dict[str, float | int | datetime]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not (
                row.get("batch_size")
                and row.get("latency_mean_ms")
                and row.get("images_per_sec")
            ):
                continue
            try:
                rows.append(
                    {
                        "batch": int(row["batch_size"]),
                        "latency": float(row["latency_mean_ms"]),
                        "ips": float(row["images_per_sec"]),
                        "warmup": int(row.get("warmup") or 0),
                        "iters": int(row.get("iters") or 0),
                        "timestamp": datetime.fromisoformat(row["timestamp"]),
                    }
                )
            except (TypeError, ValueError):
                continue

    if not rows:
        raise RuntimeError(f"No usable rows in {path}")

    comparable = [r for r in rows if r["warmup"] == 5 and r["iters"] == 20]
    rows = comparable or rows

    latest_by_batch: dict[int, dict[str, float | int | datetime]] = {}
    for row in rows:
        batch = int(row["batch"])
        old = latest_by_batch.get(batch)
        if old is None or row["timestamp"] > old["timestamp"]:
            latest_by_batch[batch] = row

    return [latest_by_batch[batch] for batch in sorted(latest_by_batch)]


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    summary_lines = []

    for label, path, color, marker in SERIES:
        data = read_latest_sweep(path)
        batches = [int(d["batch"]) for d in data]
        latencies = [float(d["latency"]) for d in data]
        img_per_sec = [float(d["ips"]) for d in data]

        axes[0].plot(
            batches,
            latencies,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            color=color,
            label=label,
        )
        axes[1].plot(
            batches,
            img_per_sec,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            color=color,
            label=label,
        )

        best = max(data, key=lambda d: float(d["ips"]))
        summary_lines.append(
            f"{label}: best {float(best['ips']):.1f} img/s @ batch "
            f"{int(best['batch'])}, batch1 {float(data[0]['latency']):.1f} ms"
        )
        axes[1].annotate(
            f"{float(best['ips']):.1f}",
            xy=(int(best["batch"]), float(best["ips"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=9,
        )

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
        ax.grid(True, which="major", color="#d8dee9", linewidth=0.8)
        ax.grid(True, which="minor", color="#edf2f7", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("Batch size")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Latency mean (ms, log scale)")
    axes[0].set_title("DINOv3 latency by batch size")
    axes[1].set_ylabel("Images / second")
    axes[1].set_title("DINOv3 throughput by batch size")
    axes[1].set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        "DINOv3 ViT-7B/16 speed comparison on RTX 5090 "
        "(256x256, warmup=5, iters=20)",
        y=1.12,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(0.01, -0.03, " | ".join(summary_lines), fontsize=9, color="#495057")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
