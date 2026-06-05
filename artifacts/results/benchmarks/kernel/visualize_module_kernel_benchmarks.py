#!/usr/bin/env python
"""Visualize module-forward 5-kernel benchmark results.

Reads:
  module_fix_m.csv
  module_fix_n.csv
  module_fix_k.csv

Writes:
  module_heatmap_fix_m.png
  module_heatmap_fix_n.png
  module_heatmap_fix_k.png
  module_speedup_fix_m.png
  module_speedup_fix_n.png
  module_speedup_fix_k.png
  module_roofline.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


KERNELS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]
KERNEL_LABELS = {
    "dense_bf16": "BF16",
    "dense_nvfp4": "N4",
    "sparse_bf16": "SB16",
    "sparse_nvfp4": "SN4",
    "marlin_nvfp4": "M4",
}
KERNEL_COLORS = {
    "dense_bf16": "#3498DB",
    "dense_nvfp4": "#27AE60",
    "sparse_bf16": "#E67E22",
    "sparse_nvfp4": "#E74C3C",
    "marlin_nvfp4": "#8E44AD",
}
KERNEL_CMAP_IDX = {k: i for i, k in enumerate(KERNELS)}

FIXED_VALUES = [1, 16, 64, 256, 4096, 16384]
POWERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
DIM_LABELS = {"m": "M (tokens)", "n": "N (out_features)", "k": "K (in_features)"}
OUTPUT_DIR = Path(__file__).resolve().parent


def load_all_data() -> list[dict]:
    rows: list[dict] = []
    for dim in ["m", "n", "k"]:
        path = OUTPUT_DIR / f"module_fix_{dim}.csv"
        if path.exists():
            with path.open(newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def _varying_dims(fixed_dim: str) -> tuple[str, str]:
    if fixed_dim == "m":
        return "n", "k"
    if fixed_dim == "n":
        return "m", "k"
    return "m", "n"


def _latency_maps(rows: list[dict]) -> tuple[dict[tuple, dict[str, float]], dict[tuple, str], dict[tuple, float]]:
    latencies: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r.get("status") != "pass" or not r.get("latency_ms") or r.get("kernel") not in KERNELS:
            continue
        key = (r["fixed_dim"], int(r["fixed_value"]), int(r["m"]), int(r["n"]), int(r["k"]))
        latencies[key][r["kernel"]] = float(r["latency_ms"])

    best_kernel: dict[tuple, str] = {}
    best_speedup: dict[tuple, float] = {}
    for key, kerns in latencies.items():
        if not kerns:
            continue
        best = min(kerns, key=kerns.get)
        best_kernel[key] = best
        dense = kerns.get("dense_bf16")
        if dense and kerns[best] > 0:
            best_speedup[key] = dense / kerns[best]
    return latencies, best_kernel, best_speedup


def plot_best_kernel_heatmaps(rows: list[dict]) -> None:
    _, best_kernel, _ = _latency_maps(rows)

    for fixed_dim in ["m", "n", "k"]:
        x_dim, y_dim = _varying_dims(fixed_dim)
        fig, axes = plt.subplots(2, 3, figsize=(28, 16))
        fig.suptitle(
            f"Fastest Packaged Linear Module -- Fixed {DIM_LABELS[fixed_dim]}   (x={x_dim.upper()}, y={y_dim.upper()})",
            fontsize=18,
            fontweight="bold",
            y=0.99,
        )
        for idx, fv in enumerate(FIXED_VALUES):
            ax = axes[idx // 3][idx % 3]
            _draw_kernel_heatmap(ax, best_kernel, fixed_dim, fv, x_dim, y_dim)

        from matplotlib.colors import BoundaryNorm, ListedColormap

        cmap = ListedColormap([KERNEL_COLORS[k] for k in KERNELS])
        norm = BoundaryNorm(np.arange(-0.5, len(KERNELS) + 0.5, 1), len(KERNELS))
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes,
            location="right",
            fraction=0.012,
            pad=0.02,
        )
        cbar.set_ticks(range(len(KERNELS)))
        cbar.set_ticklabels([KERNEL_LABELS[k] for k in KERNELS])
        cbar.ax.tick_params(labelsize=11)

        out = OUTPUT_DIR / f"module_heatmap_fix_{fixed_dim}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {out}")


def _draw_kernel_heatmap(ax, best_kernel, fixed_dim: str, fv: int, x_dim: str, y_dim: str) -> None:
    n_vals = len(POWERS)
    matrix = np.full((n_vals, n_vals), np.nan)
    for (fd, f_val, m, n, k), kernel in best_kernel.items():
        if fd != fixed_dim or f_val != fv:
            continue
        vals = {"m": m, "n": n, "k": k}
        x_val = vals[x_dim]
        y_val = vals[y_dim]
        if x_val in POWERS and y_val in POWERS:
            matrix[POWERS.index(y_val), POWERS.index(x_val)] = KERNEL_CMAP_IDX[kernel]

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([KERNEL_COLORS[k] for k in KERNELS])
    masked = np.ma.masked_invalid(matrix)
    ax.imshow(masked, origin="upper", aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(KERNELS) - 0.5)
    _draw_no_data(ax, matrix)
    for yi in range(n_vals):
        for xi in range(n_vals):
            val = matrix[yi, xi]
            if not np.isnan(val):
                kernel = KERNELS[int(val)]
                ax.text(
                    xi,
                    yi,
                    KERNEL_LABELS[kernel],
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    fontweight="bold",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=0.8, foreground="black")],
                )
    _style_grid_axes(ax, fixed_dim, fv, x_dim, y_dim)


def plot_speedup_heatmaps(rows: list[dict]) -> None:
    _, best_kernel, best_speedup = _latency_maps(rows)

    for fixed_dim in ["m", "n", "k"]:
        x_dim, y_dim = _varying_dims(fixed_dim)
        fig, axes = plt.subplots(2, 3, figsize=(28, 16))
        fig.suptitle(
            f"Best Module Speedup vs BF16 -- Fixed {DIM_LABELS[fixed_dim]}   (x={x_dim.upper()}, y={y_dim.upper()})",
            fontsize=18,
            fontweight="bold",
            y=0.99,
        )
        matrices = []
        for idx, fv in enumerate(FIXED_VALUES):
            matrix = _speedup_matrix(best_speedup, fixed_dim, fv, x_dim, y_dim)
            matrices.append(matrix)
        finite = np.concatenate([m[np.isfinite(m)] for m in matrices if np.isfinite(m).any()])
        vmax = float(np.nanpercentile(finite, 95)) if finite.size else 2.0
        vmax = max(vmax, 1.05)

        for idx, fv in enumerate(FIXED_VALUES):
            ax = axes[idx // 3][idx % 3]
            _draw_speedup_heatmap(ax, matrices[idx], best_kernel, fixed_dim, fv, x_dim, y_dim, vmax)

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=vmax))
        cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.012, pad=0.02)
        cbar.set_label("Speedup over BF16", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        out = OUTPUT_DIR / f"module_speedup_fix_{fixed_dim}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {out}")


def _speedup_matrix(best_speedup, fixed_dim: str, fv: int, x_dim: str, y_dim: str) -> np.ndarray:
    n_vals = len(POWERS)
    matrix = np.full((n_vals, n_vals), np.nan)
    for (fd, f_val, m, n, k), speedup in best_speedup.items():
        if fd != fixed_dim or f_val != fv:
            continue
        vals = {"m": m, "n": n, "k": k}
        x_val = vals[x_dim]
        y_val = vals[y_dim]
        if x_val in POWERS and y_val in POWERS:
            matrix[POWERS.index(y_val), POWERS.index(x_val)] = speedup
    return matrix


def _draw_speedup_heatmap(ax, matrix, best_kernel, fixed_dim: str, fv: int, x_dim: str, y_dim: str, vmax: float) -> None:
    masked = np.ma.masked_invalid(matrix)
    ax.imshow(masked, origin="upper", aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
    _draw_no_data(ax, matrix)
    for yi in range(len(POWERS)):
        for xi in range(len(POWERS)):
            speedup = matrix[yi, xi]
            if np.isnan(speedup):
                continue
            vals = {fixed_dim: fv, x_dim: POWERS[xi], y_dim: POWERS[yi]}
            key = (fixed_dim, fv, vals["m"], vals["n"], vals["k"])
            kernel = best_kernel.get(key)
            label = KERNEL_LABELS.get(kernel, "?")
            color = "white" if speedup < vmax * 0.65 else "black"
            ax.text(
                xi,
                yi,
                f"{label}\n{speedup:.1f}x",
                ha="center",
                va="center",
                fontsize=4.7,
                fontweight="bold",
                color=color,
                path_effects=[pe.withStroke(linewidth=0.6, foreground="black" if color == "white" else "white")],
            )
    _style_grid_axes(ax, fixed_dim, fv, x_dim, y_dim)


def _draw_no_data(ax, matrix: np.ndarray) -> None:
    no_data = np.isnan(matrix)
    if no_data.any():
        from matplotlib.colors import ListedColormap

        ax.imshow(
            np.where(no_data, 0, np.nan),
            origin="upper",
            aspect="auto",
            cmap=ListedColormap(["#DDDDDD"]),
            vmin=0,
            vmax=1,
            alpha=0.6,
        )


def _style_grid_axes(ax, fixed_dim: str, fv: int, x_dim: str, y_dim: str) -> None:
    tick_step = 2
    tick_pos = list(range(0, len(POWERS), tick_step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(POWERS[i]) for i in tick_pos], rotation=45, fontsize=7.5, ha="right")
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(POWERS[i]) for i in tick_pos], fontsize=7.5)
    ax.set_xlabel(f"{x_dim.upper()} ({DIM_LABELS[x_dim].split('(')[1].rstrip(')')})", fontsize=10)
    ax.set_ylabel(f"{y_dim.upper()} ({DIM_LABELS[y_dim].split('(')[1].rstrip(')')})", fontsize=10)
    ax.set_title(f"{fixed_dim.upper()}={fv}", fontsize=12, fontweight="bold")


def plot_roofline(rows: list[dict]) -> None:
    points: dict[str, list[tuple[float, float, float]]] = {k: [] for k in KERNELS}
    for r in rows:
        if r.get("status") != "pass" or not r.get("tflops") or r.get("kernel") not in KERNELS:
            continue
        m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        tflops = float(r["tflops"])
        kernel = r["kernel"]
        flops = 2.0 * m * n * k
        bytes_bf16 = 2.0 * (m * k + n * k + m * n)
        intensity = flops / max(bytes_bf16, 1.0)
        points[kernel].append((intensity, tflops, float(m * n * k)))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, threshold, title in [
        (axes[0], 1e6, "Module Forward Roofline -- shapes > 1e6"),
        (axes[1], 1e7, "Module Forward Roofline -- shapes > 1e7"),
    ]:
        for kernel in KERNELS:
            data = [(ai, tf) for ai, tf, size in points[kernel] if size > threshold]
            if not data:
                continue
            ais, tfs = zip(*data)
            ax.scatter(
                ais,
                tfs,
                c=KERNEL_COLORS[kernel],
                label=KERNEL_LABELS[kernel],
                alpha=0.35,
                s=9,
                edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Dense-equivalent Arithmetic Intensity (FLOP / BF16 Byte)", fontsize=12)
        ax.set_ylabel("Dense-equivalent TFLOPS", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right", markerscale=2)
        ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    out = OUTPUT_DIR / "module_roofline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def print_summary(rows: list[dict]) -> None:
    print(f"loaded {len(rows)} rows")
    for dim in ["m", "n", "k"]:
        dim_rows = [r for r in rows if r.get("fixed_dim") == dim]
        print(f"fixed {dim}: {len(dim_rows)} rows")
        for kernel in KERNELS:
            kr = [r for r in dim_rows if r.get("kernel") == kernel]
            passed = sum(1 for r in kr if r.get("status") == "pass")
            skipped = sum(1 for r in kr if r.get("status") == "skip")
            errored = sum(1 for r in kr if r.get("status") == "error")
            print(f"  {kernel:13s} pass={passed:4d} skip={skipped:4d} error={errored:4d}")


def main() -> None:
    rows = load_all_data()
    print_summary(rows)
    plot_best_kernel_heatmaps(rows)
    plot_speedup_heatmaps(rows)
    plot_roofline(rows)
    print(f"done: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
