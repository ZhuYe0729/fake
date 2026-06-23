#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from common_fakevlm_pareto import DEBUG_ROOT, f, read_csv, write_csv

REPO_ROOT = DEBUG_ROOT.parents[2]
UNIFORM_ACCURACY_CSV = REPO_ROOT / "artifacts" / "debug" / "020_fakevlm_uniform_accuracy" / "summary" / "accuracy_summary.csv"
UNIFORM_SPEED_CSV = REPO_ROOT / "artifacts" / "debug" / "021_fakevlm_linear_hybrid_prefill_speed" / "speed" / "prefill_speed.csv"
TOTAL_LINEAR_LAYERS = 224

UNIFORM_METHODS = [
    ("dense_bf16", "uniform_dense_bf16", "Uniform dense BF16", {"dense_bf16": TOTAL_LINEAR_LAYERS}),
    ("dense_nvfp4", "uniform_dense_nvfp4", "Uniform dense NVFP4", {"dense_nvfp4": TOTAL_LINEAR_LAYERS}),
    ("sparse_bf16", "uniform_sparse_bf16", "Uniform sparse BF16", {"sparse_bf16": TOTAL_LINEAR_LAYERS}),
    ("sparse_nvfp4", "uniform_sparse_nvfp4", "Uniform sparse NVFP4", {"sparse_nvfp4": TOTAL_LINEAR_LAYERS}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report-ready FakeVLM Pareto plots.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--filename-suffix", default="", help="Optional suffix inserted before report output extensions.")
    parser.add_argument("--frontier-only", action="store_true", help="Hide measured mixed policies outside the measured frontier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(root / "validation" / "pareto_validation_joined.csv")
    if not rows:
        raise RuntimeError("missing joined validation rows")

    report_rows = build_report_rows(root, rows)
    suffix = normalize_suffix(args.filename_suffix)
    write_csv(suffixed_path(report_dir / "final_fakevlm_report.csv", suffix), report_rows)

    for batch in sorted({int(float(row["batch_size"])) for row in rows}):
        batch_rows = [row for row in report_rows if int(float(row["batch_size"])) == batch]
        predicted_rows = read_csv(root / "pareto" / f"batch_{batch}" / "pareto_unique_points.csv")
        plot_batch(report_dir, batch, batch_rows, predicted_rows, suffix, args.frontier_only)

    write_summary(report_dir, report_rows, suffix)
    print(f"wrote report rows={len(report_rows)} to {report_dir}")


def normalize_suffix(value: str) -> str:
    if not value:
        return ""
    return value if value.startswith("_") else f"_{value}"


def suffixed_path(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def build_report_rows(root: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    uniform_accuracy = load_uniform_accuracy()
    uniform_speed = load_uniform_speed()
    for batch in sorted({int(float(row["batch_size"])) for row in rows}):
        batch_rows = [row for row in rows if int(float(row["batch_size"])) == batch]
        dense = next(row for row in batch_rows if int(float(row["point_index"])) == 0)
        dense_ms = f(dense, "e2e_prefill_mean_ms")
        dense_acc = f(dense, "global_accuracy")
        for row in sorted(batch_rows, key=lambda item: int(float(item["point_index"]))):
            measured_ms = f(row, "e2e_prefill_mean_ms")
            accuracy = f(row, "global_accuracy")
            item = {
                "row_type": "pareto",
                "label": f"batch_{batch}_point_{int(float(row['point_index'])):03d}",
                "batch_size": batch,
                "point_index": int(float(row["point_index"])),
                "fakeclue_accuracy": f"{accuracy:.8f}",
                "accuracy_delta_vs_dense": f"{accuracy - dense_acc:.8f}",
                "predicted_quality_cost": row["quality_cost"],
                "predicted_accuracy_from_cost": f"{dense_acc - f(row, 'quality_cost'):.8f}",
                "e2e_prefill_mean_ms": row["e2e_prefill_mean_ms"],
                "e2e_speedup_vs_dense": f"{dense_ms / measured_ms:.8f}",
                "predicted_linear_latency_ms": row["latency_ms"],
                "predicted_linear_speedup": row["speedup_vs_dense_linear"],
                "count_dense_bf16": row["count_dense_bf16"],
                "count_dense_nvfp4": row["count_dense_nvfp4"],
                "count_sparse_bf16": row["count_sparse_bf16"],
                "count_sparse_nvfp4": row["count_sparse_nvfp4"],
                "policy_json": row["policy_json"],
                "source": "024_fakevlm_prefill_global_pareto",
            }
            out.append(item)
        out.extend(build_uniform_rows(batch, dense_ms, dense_acc, uniform_accuracy, uniform_speed))
    return out


def load_uniform_accuracy() -> dict[str, float]:
    rows = read_csv(UNIFORM_ACCURACY_CSV)
    return {row["method"]: f(row, "global_accuracy") for row in rows if row.get("status", "ok") == "ok"}


def load_uniform_speed() -> dict[tuple[int, str], dict[str, str]]:
    rows = read_csv(UNIFORM_SPEED_CSV)
    best = {}
    for row in rows:
        family = row["family"]
        if not family.startswith("uniform_"):
            continue
        key = (int(float(row["batch_size"])), family)
        previous = best.get(key)
        if previous is None or speed_row_rank(row) > speed_row_rank(previous):
            best[key] = row
    return best


def speed_row_rank(row: dict[str, str]) -> tuple[int, str]:
    return (int(float(row.get("iters") or 0)), row.get("timestamp", ""))


def build_uniform_rows(
    batch: int,
    dense_ms: float,
    dense_acc: float,
    uniform_accuracy: dict[str, float],
    uniform_speed: dict[tuple[int, str], dict[str, str]],
) -> list[dict[str, str]]:
    out = []
    for index, (method, family, label, counts) in enumerate(UNIFORM_METHODS, start=101):
        acc = uniform_accuracy[method]
        if method == "dense_bf16":
            measured_ms = dense_ms
            speed_source = "024_dense_reference"
        else:
            speed_row = uniform_speed[(batch, family)]
            measured_ms = f(speed_row, "latency_mean_ms")
            speed_source = "021_fakevlm_linear_hybrid_prefill_speed"
        row = {
            "row_type": "uniform",
            "label": label,
            "batch_size": batch,
            "point_index": index,
            "fakeclue_accuracy": f"{acc:.8f}",
            "accuracy_delta_vs_dense": f"{acc - dense_acc:.8f}",
            "predicted_quality_cost": "",
            "predicted_accuracy_from_cost": "",
            "e2e_prefill_mean_ms": f"{measured_ms:.8f}",
            "e2e_speedup_vs_dense": f"{dense_ms / measured_ms:.8f}",
            "predicted_linear_latency_ms": "",
            "predicted_linear_speedup": "",
            "count_dense_bf16": str(counts.get("dense_bf16", 0)),
            "count_dense_nvfp4": str(counts.get("dense_nvfp4", 0)),
            "count_sparse_bf16": str(counts.get("sparse_bf16", 0)),
            "count_sparse_nvfp4": str(counts.get("sparse_nvfp4", 0)),
            "policy_json": "",
            "source": f"accuracy=020_fakevlm_uniform_accuracy;speed={speed_source}",
        }
        out.append(row)
    return out


def measured_frontier(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (-f(row, "e2e_speedup_vs_dense"), -f(row, "fakeclue_accuracy")))
    frontier = []
    best_acc = -1.0
    for row in ordered:
        acc = f(row, "fakeclue_accuracy")
        if acc > best_acc + 1e-12:
            frontier.append(row)
            best_acc = acc
    return list(reversed(frontier))


def plot_batch(
    report_dir: Path,
    batch: int,
    rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
    suffix: str,
    frontier_only: bool,
) -> None:
    pareto_rows = [row for row in rows if row["row_type"] == "pareto"]
    uniform_rows = [row for row in rows if row["row_type"] == "uniform"]
    dense = next(row for row in pareto_rows if int(float(row["point_index"])) == 0)
    dense_acc = f(dense, "fakeclue_accuracy")

    fig, ax = plt.subplots(figsize=(11.1, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = [f(row, "e2e_speedup_vs_dense") for row in pareto_rows]
    ys = [f(row, "fakeclue_accuracy") for row in pareto_rows]
    if not frontier_only:
        ax.plot(
            xs,
            ys,
            color="#94a3b8",
            linewidth=1.2,
            marker="o",
            markersize=5,
            markerfacecolor="#94a3b8",
            markeredgecolor="#64748b",
            label="Measured mixed policies",
            zorder=2,
        )

    frontier = measured_frontier(pareto_rows)
    ax.plot(
        [f(row, "e2e_speedup_vs_dense") for row in frontier],
        [f(row, "fakeclue_accuracy") for row in frontier],
        color="#111827",
        linewidth=2.6,
        marker="o",
        markersize=7,
        label="Measured mixed Pareto frontier",
        zorder=4,
    )

    if not frontier_only:
        for row in pareto_rows:
            point = int(float(row["point_index"]))
            if point not in {19, 20, 21}:
                continue
            offsets = {19: ((0, 9), "center"), 20: ((-10, 14), "right"), 21: ((7, -18), "left")}
            offset, horizontal_alignment = offsets[point]
            ax.annotate(
                f"P{point}",
                (f(row, "e2e_speedup_vs_dense"), f(row, "fakeclue_accuracy")),
                xytext=offset,
                textcoords="offset points",
                ha=horizontal_alignment,
                fontsize=8.5,
                color="#334155",
                zorder=6,
            )

    plot_uniform_baselines(ax, uniform_rows)

    ax.axhline(dense_acc, color="#111827", linewidth=1.0, linestyle=":", alpha=0.75, label="Dense BF16 accuracy")
    ax.axvline(1.0, color="#111827", linewidth=1.0, linestyle=":", alpha=0.75)

    ax.set_title(f"FakeVLM Prefill Pareto vs Uniform Baselines: Batch Size {batch}", fontsize=16, pad=12)
    ax.set_xlabel("Measured E2E prefill speedup vs dense BF16 (higher is better)", fontsize=12)
    ax.set_ylabel("FakeClue global accuracy (higher is better)", fontsize=12)
    ax.grid(True, which="major", color="#e5e7eb", linewidth=0.8)
    ax.grid(True, which="minor", color="#f3f4f6", linewidth=0.5)
    ax.minorticks_on()

    uniform_x = [f(row, "e2e_speedup_vs_dense") for row in uniform_rows]
    uniform_y = [f(row, "fakeclue_accuracy") for row in uniform_rows]
    x_min = min(xs + uniform_x)
    x_max = max(xs + uniform_x)
    y_min = min(ys + uniform_y)
    y_max = max(ys + uniform_y)
    ax.set_xlim(max(0.95, x_min - 0.03), x_max + 0.08)
    ax.set_ylim(max(0.70, y_min - 0.015), min(1.0, y_max + 0.012))

    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=10)
    fig.tight_layout()

    png = suffixed_path(report_dir / f"pareto_batch_{batch}_speed_vs_fakeclue.png", suffix)
    pdf = suffixed_path(report_dir / f"pareto_batch_{batch}_speed_vs_fakeclue.pdf", suffix)
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)


def plot_uniform_baselines(ax, rows: list[dict[str, str]]) -> None:
    styles = {
        "Uniform dense BF16": ("dense_bf16", (7, -18)),
        "Uniform dense NVFP4": ("dense_nvfp4", (7, -18)),
        "Uniform sparse BF16": ("sparse_bf16", (7, 10)),
        "Uniform sparse NVFP4": ("sparse_nvfp4", (7, -18)),
    }
    for row in rows:
        short_label, offset = styles[row["label"]]
        x = f(row, "e2e_speedup_vs_dense")
        y = f(row, "fakeclue_accuracy")
        ax.scatter(
            [x],
            [y],
            marker="s",
            s=104,
            color="#dc2626",
            edgecolor="white",
            linewidth=1.0,
            label="Uniform baselines" if row is rows[0] else None,
            zorder=5,
        )
        ax.annotate(
            short_label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            color="#b91c1c",
            zorder=6,
        )


def label_offset(point: int) -> tuple[int, int]:
    offsets = {
        0: (-24, -16),
        5: (5, 10),
        9: (5, 12),
        13: (5, 12),
        17: (5, 12),
        18: (5, 12),
        21: (6, 8),
        22: (6, 8),
        25: (6, 8),
        26: (6, 8),
        29: (6, 8),
        30: (6, -18),
    }
    return offsets.get(point, (6, 8))


def write_summary(report_dir: Path, rows: list[dict[str, str]], suffix: str) -> None:
    lines = [
        "# FakeVLM Prefill Pareto Report",
        "",
        "Pareto rows are selected mixed policies from `024_fakevlm_prefill_global_pareto` with full FakeClue validation and measured E2E prefill latency.",
        "Uniform rows use FakeClue accuracy from `020_fakevlm_uniform_accuracy`; non-dense uniform speed uses measured prefill latency from `021_fakevlm_linear_hybrid_prefill_speed`.",
        "",
        "## Files",
        "",
        "- `final_fakevlm_report.csv`",
    ]
    for batch in sorted({int(float(row["batch_size"])) for row in rows}):
        lines.append(f"- `pareto_batch_{batch}_speed_vs_fakeclue.png`")
        lines.append(f"- `pareto_batch_{batch}_speed_vs_fakeclue.pdf`")
    lines.extend(["", "## Batch Summary", "", "| Batch | Dense ms | Fastest point | Fastest speedup | Fastest acc | Best acc point | Best acc |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for batch in sorted({int(float(row["batch_size"])) for row in rows}):
        batch_rows = [row for row in rows if int(float(row["batch_size"])) == batch and row["row_type"] == "pareto"]
        dense = next(row for row in batch_rows if int(float(row["point_index"])) == 0)
        fastest = max(batch_rows, key=lambda row: f(row, "e2e_speedup_vs_dense"))
        best_acc = max(batch_rows, key=lambda row: f(row, "fakeclue_accuracy"))
        lines.append(
            f"| {batch} | {f(dense, 'e2e_prefill_mean_ms'):.3f} | P{int(float(fastest['point_index'])):02d} | "
            f"{f(fastest, 'e2e_speedup_vs_dense'):.3f} | {f(fastest, 'fakeclue_accuracy'):.4f} | "
            f"P{int(float(best_acc['point_index'])):02d} | {f(best_acc, 'fakeclue_accuracy'):.4f} |"
        )
    lines.extend(["", "## Selected Points", "", "| Batch | Point | Speedup | E2E ms | Accuracy | Replaced | Counts |", "|---:|---:|---:|---:|---:|---:|---|"])
    for row in rows:
        if row["row_type"] != "pareto":
            continue
        counts = ", ".join(
            f"{name}={row[f'count_{name}']}"
            for name in ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
            if row[f"count_{name}"] not in {"0", "0.0", ""}
        )
        replaced = TOTAL_LINEAR_LAYERS - int(float(row["count_dense_bf16"]))
        lines.append(
            f"| {row['batch_size']} | {row['point_index']} | {f(row, 'e2e_speedup_vs_dense'):.3f} | "
            f"{f(row, 'e2e_prefill_mean_ms'):.3f} | {f(row, 'fakeclue_accuracy'):.4f} | {replaced} | {counts} |"
        )
    lines.extend(["", "## Uniform Baselines", "", "| Batch | Method | Speedup | E2E ms | Accuracy | Counts |", "|---:|---|---:|---:|---:|---|"])
    for row in rows:
        if row["row_type"] != "uniform":
            continue
        counts = ", ".join(
            f"{name}={row[f'count_{name}']}"
            for name in ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
            if row[f"count_{name}"] not in {"0", "0.0", ""}
        )
        lines.append(
            f"| {row['batch_size']} | {row['label']} | {f(row, 'e2e_speedup_vs_dense'):.3f} | "
            f"{f(row, 'e2e_prefill_mean_ms'):.3f} | {f(row, 'fakeclue_accuracy'):.4f} | {counts} |"
        )
    suffixed_path(report_dir / "final_report_summary.md", suffix).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
