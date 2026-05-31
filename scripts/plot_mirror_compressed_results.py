#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


METHOD_ORDER = [
    "dense",
    "nvfp4",
    "int4",
    "unstructured_sparse",
    "semi_structured_sparse",
    "nvfp4_unstructured_sparse",
    "nvfp4_semi_structured_sparse",
    "int4_unstructured_sparse",
    "int4_semi_structured_sparse",
    "nvfp4_4over6_unstructured_sparse",
    "nvfp4_4over6_semi_structured_sparse",
]
RUNTIME_METHODS = ["dense", "semi_structured_sparse", "nvfp4", "nvfp4_semi_structured_sparse"]
PLOT_METHOD_ORDER = [method for method in METHOD_ORDER if "int4" not in method]
GENIMAGE_DATASETS = [
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "ADM",
    "glide",
    "wukong",
    "VQDM",
    "BigGAN",
]
METHOD_LABELS = {
    "dense": "Dense",
    "nvfp4": "NVFP4",
    "int4": "INT4",
    "unstructured_sparse": "Unstructured",
    "semi_structured_sparse": "2:4 Sparse",
    "nvfp4_unstructured_sparse": "NVFP4 + Unstruct",
    "nvfp4_semi_structured_sparse": "NVFP4 + 2:4",
    "int4_unstructured_sparse": "INT4 + Unstruct",
    "int4_semi_structured_sparse": "INT4 + 2:4",
    "nvfp4_4over6_unstructured_sparse": "4/6 NVFP4 + Unstruct",
    "nvfp4_4over6_semi_structured_sparse": "4/6 NVFP4 + 2:4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot MIRROR compressed evaluation results.")
    parser.add_argument("--accuracy-csv", default="artifacts/results/mirror_compressed/accuracy.csv")
    parser.add_argument("--speed-csv", default="artifacts/results/mirror_compressed/speed.csv")
    parser.add_argument("--output-dir", default="artifacts/results/mirror_compressed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracy_rows = _latest_rows(_read_csv(Path(args.accuracy_csv)), ["method", "benchmark", "dataset"])
    speed_rows = _latest_rows(_read_csv(Path(args.speed_csv)), ["method"])
    summary_rows = _build_summary(accuracy_rows, speed_rows)
    summary_path = output_dir / "summary.csv"
    _write_csv(summary_path, summary_rows)
    _plot_accuracy_summary(summary_rows, output_dir / "accuracy_summary.png")
    _plot_genimage_breakdown(accuracy_rows, output_dir / "genimage_breakdown.png")
    _plot_speed_summary(summary_rows, output_dir / "speed_summary.png")
    print(f"wrote {summary_path}")
    print(f"wrote {output_dir / 'accuracy_summary.png'}")
    print(f"wrote {output_dir / 'genimage_breakdown.png'}")
    print(f"wrote {output_dir / 'speed_summary.png'}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _latest_rows(rows: list[dict[str, str]], keys: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    latest: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in keys)
        if key not in latest or row.get("timestamp", "") >= latest[key].get("timestamp", ""):
            latest[key] = row
    return latest


def _build_summary(
    accuracy_rows: dict[tuple[str, ...], dict[str, str]],
    speed_rows: dict[tuple[str, ...], dict[str, str]],
) -> list[dict[str, object]]:
    dense_speed = _float(speed_rows.get(("dense",), {}).get("images_per_sec", ""))
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        chameleon = accuracy_rows.get((method, "Chameleon", "ALL"), {})
        gen_mean = accuracy_rows.get((method, "GenImage", "MEAN"), {})
        speed = speed_rows.get((method,), {})
        ips = _float(speed.get("images_per_sec", ""))
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "chameleon_bal_acc": _fmt(_float(chameleon.get("bal_acc", ""))),
                "chameleon_auc": _fmt(_float(chameleon.get("auc", ""))),
                "genimage_mean_bal_acc": _fmt(_float(gen_mean.get("bal_acc", ""))),
                "genimage_mean_auc": _fmt(_float(gen_mean.get("auc", ""))),
                "genimage_biggan_bal_acc": _fmt(
                    _float(accuracy_rows.get((method, "GenImage", "BigGAN"), {}).get("bal_acc", ""))
                ),
                "speed_latency_mean_ms": _fmt(_float(speed.get("latency_mean_ms", ""))),
                "speed_images_per_sec": _fmt(ips),
                "speedup_vs_dense": _fmt(ips / dense_speed if dense_speed and ips else math.nan),
                "kernel_backend": speed.get("kernel_backend", ""),
                "replaced_linear_count": speed.get("replaced_linear_count", ""),
                "skipped_linear_count": speed.get("skipped_linear_count", ""),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_accuracy_summary(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plot_rows = [row for row in rows if "int4" not in str(row["method"])]
    datasets = ["Chameleon", "GenImage mean"]
    metric_keys = ["chameleon_bal_acc", "genimage_mean_bal_acc"]
    labels = [str(row["label"]) for row in plot_rows]
    values = np.array([[_float(row[key]) * 100 for row in plot_rows] for key in metric_keys], dtype=float)
    x = np.arange(len(datasets))
    width = min(0.095, 0.82 / max(1, len(plot_rows)))
    offsets = (np.arange(len(plot_rows)) - (len(plot_rows) - 1) / 2) * width
    colors = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=(14.8, 6.0))
    for method_idx, label in enumerate(labels):
        bars = ax.bar(
            x + offsets[method_idx],
            values[:, method_idx],
            width,
            label=label,
            color=colors[method_idx % len(colors)],
        )
        for bar, value in zip(bars, values[:, method_idx]):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.45,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(max(0, float(np.nanmin(values)) - 4.0), 103.0)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncols=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_genimage_breakdown(
    accuracy_rows: dict[tuple[str, ...], dict[str, str]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(
        [
            [_float(accuracy_rows.get((method, "GenImage", dataset), {}).get("bal_acc", "")) * 100 for dataset in GENIMAGE_DATASETS]
            for method in PLOT_METHOD_ORDER
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    vmin = max(50.0, float(np.nanmin(matrix)) - 2.0)
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=100.0)
    ax.set_xticks(range(len(GENIMAGE_DATASETS)))
    ax.set_xticklabels(GENIMAGE_DATASETS, rotation=28, ha="right")
    ax.set_yticks(range(len(PLOT_METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS.get(method, method) for method in PLOT_METHOD_ORDER])
    ax.set_title("GenImage Balanced Accuracy (%)")
    ax.set_xticks(np.arange(-0.5, len(GENIMAGE_DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PLOT_METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            r, g, b, _ = image.cmap(image.norm(matrix[i, j]))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#1F2933" if luminance > 0.62 else "white"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=9)
    fig.colorbar(image, ax=ax, label="Balanced accuracy (%)", fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_speed_summary(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    runtime_rows = [row for row in rows if row["method"] in RUNTIME_METHODS and row["speed_images_per_sec"]]
    labels = [str(row["label"]) for row in runtime_rows]
    ips = [_float(row["speed_images_per_sec"]) for row in runtime_rows]
    latency = [_float(row["speed_latency_mean_ms"]) for row in runtime_rows]
    speedup = [_float(row["speedup_vs_dense"]) for row in runtime_rows]
    x = list(range(len(runtime_rows)))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(x, ips, color="#54A24B")
    axes[0].set_ylabel("Images / sec")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.8)
    for i, value in enumerate(speedup):
        axes[0].text(i, ips[i], f"{value:.2f}x", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, latency, color="#E45756")
    axes[1].set_ylabel("Latency mean (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _float(value: object) -> float:
    try:
        if value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


if __name__ == "__main__":
    main()
