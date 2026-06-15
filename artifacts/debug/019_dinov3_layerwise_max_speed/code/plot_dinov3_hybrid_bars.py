#!/usr/bin/env python
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (parent for parent in (CODE_DIR, *CODE_DIR.parents) if (parent / "fake").is_dir() and (parent / "artifacts").is_dir()),
    CODE_DIR.parents[3],
)
OUT_DIR = REPO_ROOT / "artifacts/debug/019_dinov3_layerwise_max_speed/hybrid_vs_uniform"

SERIES = [
    {
        "key": "dense_fp32",
        "label": "Dense FP32",
        "path": REPO_ROOT / "artifacts/results/dinov3_vit7b16_dense/speed.csv",
        "color": "#4c6ef5",
    },
    {
        "key": "dense_nvfp4",
        "label": "Dense NVFP4",
        "path": REPO_ROOT / "artifacts/results/dinov3_vit7b16_cutlass_nvfp4/speed.csv",
        "color": "#f08c00",
    },
    {
        "key": "sparse_bf16",
        "label": "Sparse BF16",
        "path": REPO_ROOT / "artifacts/results/dinov3_vit7b16_cutlass_sparse_bf16/speed.csv",
        "color": "#12b886",
    },
    {
        "key": "sparse_nvfp4",
        "label": "Sparse NVFP4",
        "path": REPO_ROOT / "artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/speed_storage.csv",
        "color": "#15aabf",
    },
    {
        "key": "hybrid",
        "label": "Hybrid",
        "path": REPO_ROOT / "artifacts/results/dinov3_vit7b16_cutlass_hybrid/speed.csv",
        "color": "#e03131",
    },
]
BATCHES = [32]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    write_csv(OUT_DIR / "hybrid_vs_uniform_summary.csv", rows)
    plot_metric(rows, "images_per_sec", "Images / sec", OUT_DIR / "hybrid_vs_uniform_throughput.png")
    plot_metric(rows, "latency_mean_ms", "Mean latency (ms)", OUT_DIR / "hybrid_vs_uniform_latency.png")
    write_readme(rows)
    print(OUT_DIR / "hybrid_vs_uniform_throughput.png")
    print(OUT_DIR / "hybrid_vs_uniform_latency.png")
    print(OUT_DIR / "hybrid_vs_uniform_summary.csv")


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_series = {item["key"]: latest_by_batch(item["path"]) for item in SERIES}
    for batch in BATCHES:
        dense_ips = _float(by_series["dense_fp32"].get(batch, {}).get("images_per_sec"))
        best_uniform_ips = max(
            _float(by_series[key].get(batch, {}).get("images_per_sec"))
            for key in ("dense_fp32", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
        )
        for item in SERIES:
            source = by_series[item["key"]].get(batch)
            if source is None:
                continue
            ips = _float(source.get("images_per_sec"))
            scheme = source.get("hybrid_scheme", "")
            if scheme:
                scheme = scheme.split("_", 1)[0]
            row = {
                "batch_size": batch,
                "method_key": item["key"],
                "method_label": item["label"],
                "hybrid_scheme": scheme,
                "latency_mean_ms": source.get("latency_mean_ms", ""),
                "images_per_sec": source.get("images_per_sec", ""),
                "speedup_vs_dense_fp32": "" if dense_ips <= 0 else f"{ips / dense_ips:.6f}",
                "speedup_vs_best_uniform": "" if best_uniform_ips <= 0 else f"{ips / best_uniform_ips:.6f}",
                "warmup": source.get("warmup", ""),
                "iters": source.get("iters", ""),
                "timestamp": source.get("timestamp", ""),
                "source_csv": str(item["path"].relative_to(REPO_ROOT)),
            }
            rows.append(row)
    return rows


def latest_by_batch(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                batch = int(row.get("batch_size", ""))
                datetime.fromisoformat(row.get("timestamp", ""))
            except ValueError:
                continue
            old = rows.get(batch)
            if old is None or row.get("timestamp", "") >= old.get("timestamp", ""):
                rows[batch] = row
    return rows


def plot_metric(rows: list[dict[str, Any]], metric: str, ylabel: str, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "figure.dpi": 170,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    width = 0.13
    x_positions = list(range(len(BATCHES)))
    offsets = [(-2 + index) * width for index in range(len(SERIES))]
    rows_by_key_batch = {(row["method_key"], int(row["batch_size"])): row for row in rows}

    for index, item in enumerate(SERIES):
        values = []
        for batch in BATCHES:
            row = rows_by_key_batch.get((item["key"], batch), {})
            value = _float(row.get(metric))
            values.append(value)
        xs = [pos + offsets[index] for pos in x_positions]
        bars = ax.bar(xs, values, width=width, label=item["label"], color=item["color"])
        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            annotation = f"{value:.1f}"
            ax.annotate(
                annotation,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="medium",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Batch {batch}" for batch in BATCHES])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#e9ecef", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#495057")
    ax.spines["bottom"].set_color("#495057")
    upper = max(_float(row.get(metric)) for row in rows)
    ax.set_ylim(0, upper * 1.18 if upper > 0 else 1)
    fig.suptitle("DINOv3 Hybrid vs Uniform Methods", y=0.97, fontsize=16, fontweight="medium")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=len(SERIES),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.98)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_readme(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DINOv3 Hybrid Comparison",
        "",
        "Compares the existing hybrid run against single uniform methods at the complete overlapping batch size.",
        "",
        "The hybrid result improves batch-32 throughput from the best uniform result, Sparse BF16 at 81.607 images/sec, to 86.362 images/sec. That is a 1.058x speedup over the best uniform DINOv3 baseline.",
        "",
        "This gain is much smaller than the LLaMA-2 prefill-only hybrid results because DINOv3 has less exploitable layer-to-layer kernel diversity in the measured setup. The DINOv3 transformer blocks repeat the same small set of ViT projection shapes, and at batch 32 a single uniform backend, Sparse BF16, is already close to optimal for most of the runtime. The existing DINOv3 hybrid only switches part of the model to the faster backend, so it mostly captures a small residual improvement.",
        "",
        "In the earlier LLaMA-2 prefill-only study, the mixed policies had a much wider useful design space: different projection groups had different best backends and the Pareto-selected policies combined Dense BF16, Dense NVFP4, and Sparse BF16. The recorded LLaMA-2 prefill points reached about 1.49x to 1.64x speedup versus dense while also outperforming the uniform sparse baselines in quality-sensitive comparisons. That larger spread leaves more room for hybrid routing to beat any one uniform method.",
        "",
        "| Batch | Method | img/s | latency ms | speedup vs dense | speedup vs best uniform |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['method_label']} | {row['images_per_sec']} | "
            f"{row['latency_mean_ms']} | {row['speedup_vs_dense_fp32']} | {row['speedup_vs_best_uniform']} |"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
