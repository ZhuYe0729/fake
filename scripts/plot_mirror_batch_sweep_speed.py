#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


METHOD_ORDER = ["dense", "semi_structured_sparse", "nvfp4", "nvfp4_semi_structured_sparse"]
METHOD_LABELS = {
    "dense": "Dense",
    "semi_structured_sparse": "2:4 Sparse BF16",
    "nvfp4": "NVFP4",
    "nvfp4_semi_structured_sparse": "Sparse NVFP4",
}
COLORS = {
    "dense": "#4C78A8",
    "semi_structured_sparse": "#54A24B",
    "nvfp4": "#F58518",
    "nvfp4_semi_structured_sparse": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MIRROR runtime batch sweep speed results.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "artifacts/results/mirror_compressed/speed_batch_sweep_dense.csv",
            "artifacts/results/mirror_compressed/speed_batch_sweep_semi_structured_sparse.csv",
            "artifacts/results/mirror_compressed/speed_batch_sweep_nvfp4.csv",
            "artifacts/results/mirror_compressed/speed_batch_sweep_nvfp4_semi_structured_sparse.csv",
        ],
    )
    parser.add_argument("--output-dir", default="artifacts/results/mirror_compressed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_rows([Path(path) for path in args.inputs])
    rows = _summary_rows(latest)
    summary_path = output_dir / "speed_batch_sweep_summary.csv"
    _write_csv(summary_path, rows)
    _plot_metric(rows, "images_per_sec", "Images / sec", output_dir / "speed_batch_sweep_throughput.png")
    _plot_metric(rows, "latency_mean_ms", "Latency mean (ms)", output_dir / "speed_batch_sweep_latency.png")
    _plot_metric(rows, "speedup_vs_dense", "Speedup vs dense", output_dir / "speed_batch_sweep_speedup.png")
    print(f"wrote {summary_path}")
    print(f"wrote {output_dir / 'speed_batch_sweep_throughput.png'}")
    print(f"wrote {output_dir / 'speed_batch_sweep_latency.png'}")
    print(f"wrote {output_dir / 'speed_batch_sweep_speedup.png'}")


def _latest_rows(paths: list[Path]) -> dict[tuple[str, int], dict[str, str]]:
    latest: dict[tuple[str, int], dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                method = row.get("method", "")
                batch_size = int(row.get("batch_size", "0"))
                key = (method, batch_size)
                if key not in latest or row.get("timestamp", "") >= latest[key].get("timestamp", ""):
                    latest[key] = row
    return latest


def _summary_rows(latest: dict[tuple[str, int], dict[str, str]]) -> list[dict[str, object]]:
    dense_ips = {
        batch_size: _float(row.get("images_per_sec", ""))
        for (method, batch_size), row in latest.items()
        if method == "dense"
    }
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_batches = sorted(batch for current_method, batch in latest if current_method == method)
        for batch_size in method_batches:
            row = latest[(method, batch_size)]
            ips = _float(row.get("images_per_sec", ""))
            dense_ref = dense_ips.get(batch_size, math.nan)
            rows.append(
                {
                    "method": method,
                    "label": METHOD_LABELS.get(method, method),
                    "batch_size": batch_size,
                    "latency_mean_ms": _fmt(_float(row.get("latency_mean_ms", ""))),
                    "latency_p50_ms": _fmt(_float(row.get("latency_p50_ms", ""))),
                    "latency_p90_ms": _fmt(_float(row.get("latency_p90_ms", ""))),
                    "images_per_sec": _fmt(ips),
                    "speedup_vs_dense": _fmt(ips / dense_ref if ips and dense_ref else math.nan),
                    "runtime_dtype": row.get("runtime_dtype", ""),
                    "kernel_backend": row.get("kernel_backend", ""),
                    "replaced_linear_count": row.get("replaced_linear_count", ""),
                    "skipped_linear_count": row.get("skipped_linear_count", ""),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows: list[dict[str, object]], metric: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method and row.get(metric, "")]
        if not method_rows:
            continue
        x = [int(row["batch_size"]) for row in method_rows]
        y = [_float(row[metric]) for row in method_rows]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.9,
            color=COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({int(row["batch_size"]) for row in rows}))
    ax.get_xaxis().set_major_formatter(lambda value, _: f"{int(value)}")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
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
