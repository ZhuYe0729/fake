#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass(frozen=True)
class SpeedSpec:
    key: str
    label: str
    panel: str
    dense_speed_csv: Path
    nvfp4_speed_csv: Path
    sparse_bf16_speed_csv: Path
    sparse_nvfp4_speed_csv: Path
    batch_size: int | None = None


@dataclass(frozen=True)
class SpeedRow:
    model_key: str
    model_label: str
    panel: str
    method: str
    method_label: str
    batch_size: int
    latency_mean_ms: float
    images_per_sec: float
    dense_images_per_sec: float
    speedup: float
    runtime_dtype: str
    source_csv: str


@dataclass(frozen=True)
class BatchSpeedRow:
    model_key: str
    method: str
    method_label: str
    batch_size: int
    latency_mean_ms: float
    images_per_sec: float
    dense_images_per_sec: float
    speedup: float
    source_csv: str


METHODS = [
    ("nvfp4", "Dense NVFP4", "#3b6ea8"),
    ("sparse_bf16", "4:8 Sparse BF16", "#b16828"),
    ("sparse_nvfp4", "4:8 Sparse NVFP4", "#0f8b8d"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot end-to-end speedup summary for PPT.")
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--csv-output", default="artifacts/results/end_to_end_speedup_summary.csv")
    parser.add_argument("--dino-batch-csv-output", default="artifacts/results/dinov3_batch_speed_summary.csv")
    parser.add_argument("--png-output", default="artifacts/results/end_to_end_speedup_summary.png")
    parser.add_argument("--pdf-output", default="artifacts/results/end_to_end_speedup_summary.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    specs = _speed_specs(results_dir)
    rows = _collect_rows(specs)
    dino_batch_rows = _collect_dino_batch_rows(_dino_spec(results_dir))
    _write_csv(rows, Path(args.csv_output))
    _write_dino_batch_csv(dino_batch_rows, Path(args.dino_batch_csv_output))
    _plot(rows, dino_batch_rows, Path(args.png_output), Path(args.pdf_output))
    print(f"[plot] wrote {args.csv_output}")
    print(f"[plot] wrote {args.dino_batch_csv_output}")
    print(f"[plot] wrote {args.png_output}")
    print(f"[plot] wrote {args.pdf_output}")


def _speed_specs(results_dir: Path) -> list[SpeedSpec]:
    specs = [
        _maxvit_spec(results_dir, "tiny", "Tiny", batch_size=128),
        _maxvit_spec(results_dir, "small", "Small", batch_size=128),
        _maxvit_spec(results_dir, "base", "Base", batch_size=128),
        _maxvit_spec(results_dir, "large", "Large", batch_size=16),
    ]
    specs.append(_dino_spec(results_dir))
    return specs


def _dino_spec(results_dir: Path) -> SpeedSpec:
    return SpeedSpec(
        key="dinov3_vit7b16",
        label="DINOv3",
        panel="dinov3",
        dense_speed_csv=results_dir / "dinov3_vit7b16_dense" / "speed.csv",
        nvfp4_speed_csv=results_dir / "dinov3_vit7b16_cutlass_nvfp4" / "speed.csv",
        sparse_bf16_speed_csv=results_dir / "dinov3_vit7b16_cutlass_sparse_bf16" / "speed.csv",
        sparse_nvfp4_speed_csv=results_dir / "dinov3_vit7b16_cutlass_sparse_nvfp4" / "speed_storage.csv",
        batch_size=8,
    )


def _maxvit_spec(results_dir: Path, variant: str, label: str, batch_size: int) -> SpeedSpec:
    return SpeedSpec(
        key=f"maxvit_{variant}",
        label=label,
        panel="maxvit",
        dense_speed_csv=results_dir / f"maxvit_{variant}_dense" / "speed.csv",
        nvfp4_speed_csv=results_dir / f"maxvit_{variant}_cutlass_nvfp4" / "speed.csv",
        sparse_bf16_speed_csv=results_dir / f"maxvit_{variant}_cutlass_sparse_bf16" / "speed.csv",
        sparse_nvfp4_speed_csv=results_dir / f"maxvit_{variant}_cutlass_sparse_nvfp4" / "speed.csv",
        batch_size=batch_size,
    )


def _collect_rows(specs: list[SpeedSpec]) -> list[SpeedRow]:
    rows: list[SpeedRow] = []
    for spec in specs:
        dense = _latest_speed_row(spec.dense_speed_csv, spec.batch_size)
        if dense is None:
            continue
        dense_ips = _float(dense.get("images_per_sec", ""))
        if dense_ips <= 0:
            continue
        method_paths = {
            "nvfp4": spec.nvfp4_speed_csv,
            "sparse_bf16": spec.sparse_bf16_speed_csv,
            "sparse_nvfp4": spec.sparse_nvfp4_speed_csv,
        }
        for method, label, _color in METHODS:
            row = _latest_speed_row(method_paths[method], spec.batch_size)
            if row is None:
                continue
            ips = _float(row.get("images_per_sec", ""))
            latency = _float(row.get("latency_mean_ms", ""))
            batch_size = int(_float(row.get("batch_size", "0")))
            if ips <= 0 or latency <= 0 or batch_size <= 0:
                continue
            rows.append(
                SpeedRow(
                    model_key=spec.key,
                    model_label=spec.label,
                    panel=spec.panel,
                    method=method,
                    method_label=label,
                    batch_size=batch_size,
                    latency_mean_ms=latency,
                    images_per_sec=ips,
                    dense_images_per_sec=dense_ips,
                    speedup=ips / dense_ips,
                    runtime_dtype=row.get("runtime_dtype", ""),
                    source_csv=str(method_paths[method]),
                )
            )
    return rows


def _collect_dino_batch_rows(spec: SpeedSpec) -> list[BatchSpeedRow]:
    dense_by_batch = _latest_speed_rows_by_batch(spec.dense_speed_csv)
    method_paths = {
        "nvfp4": spec.nvfp4_speed_csv,
        "sparse_bf16": spec.sparse_bf16_speed_csv,
        "sparse_nvfp4": spec.sparse_nvfp4_speed_csv,
    }
    rows: list[BatchSpeedRow] = []
    for method, label, _color in METHODS:
        for batch_size, method_row in sorted(_latest_speed_rows_by_batch(method_paths[method]).items()):
            dense_row = dense_by_batch.get(batch_size)
            if dense_row is None:
                continue
            dense_ips = _float(dense_row.get("images_per_sec", ""))
            ips = _float(method_row.get("images_per_sec", ""))
            latency = _float(method_row.get("latency_mean_ms", ""))
            if dense_ips <= 0 or ips <= 0 or latency <= 0:
                continue
            rows.append(
                BatchSpeedRow(
                    model_key=spec.key,
                    method=method,
                    method_label=label,
                    batch_size=batch_size,
                    latency_mean_ms=latency,
                    images_per_sec=ips,
                    dense_images_per_sec=dense_ips,
                    speedup=ips / dense_ips,
                    source_csv=str(method_paths[method]),
                )
            )
    return rows


def _write_csv(rows: list[SpeedRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "method",
                "batch_size",
                "latency_mean_ms",
                "images_per_sec",
                "dense_images_per_sec",
                "speedup",
                "runtime_dtype",
                "source_csv",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model_label,
                    "method": row.method_label,
                    "batch_size": row.batch_size,
                    "latency_mean_ms": f"{row.latency_mean_ms:.3f}",
                    "images_per_sec": f"{row.images_per_sec:.3f}",
                    "dense_images_per_sec": f"{row.dense_images_per_sec:.3f}",
                    "speedup": f"{row.speedup:.3f}",
                    "runtime_dtype": row.runtime_dtype,
                    "source_csv": row.source_csv,
                }
            )


def _write_dino_batch_csv(rows: list[BatchSpeedRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "method",
                "batch_size",
                "latency_mean_ms",
                "images_per_sec",
                "dense_images_per_sec",
                "speedup",
                "source_csv",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": "DINOv3",
                    "method": row.method_label,
                    "batch_size": row.batch_size,
                    "latency_mean_ms": f"{row.latency_mean_ms:.3f}",
                    "images_per_sec": f"{row.images_per_sec:.3f}",
                    "dense_images_per_sec": f"{row.dense_images_per_sec:.3f}",
                    "speedup": f"{row.speedup:.3f}",
                    "source_csv": row.source_csv,
                }
            )


def _plot(
    rows: list[SpeedRow],
    dino_batch_rows: list[BatchSpeedRow],
    png_output: Path,
    pdf_output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
        }
    )
    fig, (ax_maxvit, ax_dino_speedup) = plt.subplots(
        1,
        2,
        figsize=(13.8, 6.8),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.24},
    )
    method_colors = {method: color for method, _label, color in METHODS}
    _plot_panel(
        ax_maxvit,
        [row for row in rows if row.panel == "maxvit"],
        ["Tiny", "Small", "Base", "Large"],
        "MaxViT End-to-End Forward",
        x_limit_pad=0.26,
        method_colors=method_colors,
    )
    _plot_dino_batch_panel(ax_dino_speedup, dino_batch_rows, method_colors, metric="speedup")
    handles = [
        Patch(facecolor=color, edgecolor="#202020", label=label)
        for method, label, color in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.895),
        fontsize=10.8,
        handlelength=2.0,
        columnspacing=2.0,
    )
    fig.suptitle("End-to-End Throughput Speedup on RTX 5090", fontsize=15.5, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.025,
        "Speedup is relative throughput versus the dense baseline at the same batch size.  MaxViT uses batch 128 except Large batch 16; DINOv3 shows the measured batch sweep.",
        ha="center",
        fontsize=9,
        color="#4d4d4d",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.76)
    png_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)


def _plot_panel(
    ax,
    rows: list[SpeedRow],
    model_labels: list[str],
    title: str,
    x_limit_pad: float,
    method_colors: dict[str, str],
) -> None:
    by_key = {(row.model_label, row.method): row for row in rows}
    centers = list(range(len(model_labels)))
    offsets = [-0.24, 0.0, 0.24]
    bar_height = 0.20

    for offset, (method, _label, _color) in zip(offsets, METHODS, strict=True):
        values = [by_key.get((model, method)).speedup if by_key.get((model, method)) else math.nan for model in model_labels]
        y_positions = [center + offset for center in centers]
        ax.barh(
            y_positions,
            values,
            height=bar_height,
            color=method_colors[method],
            edgecolor="#202020",
            linewidth=0.7,
        )
        for y, model, value in zip(y_positions, model_labels, values, strict=True):
            row = by_key.get((model, method))
            if row is None or math.isnan(value):
                continue
            label = f"{value:.2f}x"
            ax.text(
                value + 0.035,
                y,
                label,
                va="center",
                ha="left",
                fontsize=8.4,
                color="#202020",
            )

    max_value = max((row.speedup for row in rows), default=1.0)
    ax.axvline(1.0, color="#1f1f1f", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlim(0, max_value + x_limit_pad)
    ax.set_title(title, fontsize=12.5)
    ax.set_xlabel("Throughput Speedup (x)")
    ax.set_yticks(centers, model_labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_dino_batch_panel(
    ax,
    rows: list[BatchSpeedRow],
    method_colors: dict[str, str],
    metric: str,
) -> None:
    title = "DINOv3 Speedup vs. Batch Size"
    ylabel = "Speedup vs. Dense (x)"
    for method, label, _color in METHODS:
        method_rows = sorted([row for row in rows if row.method == method], key=lambda row: row.batch_size)
        if not method_rows:
            continue
        x = [row.batch_size for row in method_rows]
        y = [row.speedup for row in method_rows]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            color=method_colors[method],
            label=label,
        )
    ax.axhline(1.0, color="#1f1f1f", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _latest_speed_row(path: Path, batch_size: int | None) -> dict[str, str] | None:
    latest: dict[str, str] | None = None
    for row in _read_csv(path):
        if batch_size is not None and row.get("batch_size") != str(batch_size):
            continue
        if latest is None or _timestamp(row) >= _timestamp(latest):
            latest = row
    return latest


def _latest_speed_rows_by_batch(path: Path) -> dict[int, dict[str, str]]:
    latest: dict[int, dict[str, str]] = {}
    for row in _read_csv(path):
        batch_size = int(_float(row.get("batch_size", "0")))
        if batch_size <= 0:
            continue
        if batch_size not in latest or _timestamp(row) >= _timestamp(latest[batch_size]):
            latest[batch_size] = row
    return latest


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _timestamp(row: dict[str, str]) -> datetime:
    try:
        return datetime.fromisoformat(row.get("timestamp", ""))
    except ValueError:
        return datetime.min


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


if __name__ == "__main__":
    main()
