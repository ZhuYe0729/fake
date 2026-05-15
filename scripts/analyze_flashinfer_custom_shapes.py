#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = "artifacts/analysis/flashinfer/custom_shapes.csv"
DEFAULT_OUTPUT_DIR = "artifacts/analysis/flashinfer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FlashInfer custom shape benchmark CSV.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(input_path)
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    if not ok_rows:
        _write_markdown(output_dir / "summary.md", input_path, [], [], [], rows)
        print(f"No OK rows found in {input_path}; wrote summary with error count only.")
        return

    shape_summaries = _shape_summaries(ok_rows)
    family_summaries = _family_summaries(shape_summaries)
    component_rows = _component_breakdown(ok_rows)

    _write_csv(output_dir / "summary.csv", shape_summaries)
    _write_csv(output_dir / "speedup_by_shape_family.csv", family_summaries)
    _write_csv(output_dir / "component_breakdown.csv", component_rows)
    _plot_outputs(output_dir, shape_summaries, component_rows)
    _write_markdown(output_dir / "summary.md", input_path, shape_summaries, family_summaries, component_rows, rows)
    print(f"analysis done: {output_dir}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _shape_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row["shape_index"], row["shape_family"], row["m"], row["n"], row["k"])


def _shape_summaries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[_shape_key(row)][row["op"]] = row

    summaries: list[dict[str, object]] = []
    for key, ops in grouped.items():
        dense_bf16 = _float_op(ops, "dense_linear_bf16")
        dense_fp32 = _float_op(ops, "dense_linear_fp32")
        nvfp4_forward = _float_op(ops, "nvfp4_forward_like")
        nvfp4_gemm = _float_op(ops, "nvfp4_gemm_only")
        quant = _float_op(ops, "activation_scale_plus_quant")
        if any(value is None for value in (dense_bf16, dense_fp32, nvfp4_forward, nvfp4_gemm, quant)):
            continue
        sample = next(iter(ops.values()))
        summaries.append(
            {
                "shape_index": int(key[0]),
                "shape_family": key[1],
                "m": int(key[2]),
                "n": int(key[3]),
                "k": int(key[4]),
                "flops": int(sample["flops"]),
                "arithmetic_intensity_bf16": _to_float(sample["arithmetic_intensity_bf16"]),
                "dense_linear_bf16_ms": dense_bf16,
                "dense_linear_fp32_ms": dense_fp32,
                "nvfp4_forward_like_ms": nvfp4_forward,
                "nvfp4_gemm_only_ms": nvfp4_gemm,
                "activation_scale_plus_quant_ms": quant,
                "forward_speedup_vs_bf16": _safe_ratio(dense_bf16, nvfp4_forward),
                "forward_speedup_vs_fp32": _safe_ratio(dense_fp32, nvfp4_forward),
                "gemm_speedup_vs_bf16": _safe_ratio(dense_bf16, nvfp4_gemm),
                "quant_share_of_forward": _safe_ratio(quant, nvfp4_forward),
                "gemm_share_of_forward": _safe_ratio(nvfp4_gemm, nvfp4_forward),
            }
        )
    summaries.sort(key=lambda row: int(row["shape_index"]))
    return summaries


def _family_summaries(shape_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in shape_rows:
        grouped[str(row["shape_family"])].append(row)
    summaries = []
    for family, rows in sorted(grouped.items()):
        speedups = [_to_float(row["forward_speedup_vs_bf16"]) for row in rows]
        quant_shares = [_to_float(row["quant_share_of_forward"]) for row in rows]
        summaries.append(
            {
                "shape_family": family,
                "shapes": len(rows),
                "accelerated_vs_bf16": sum(1 for value in speedups if value > 1.0),
                "slowed_vs_bf16": sum(1 for value in speedups if value <= 1.0),
                "median_forward_speedup_vs_bf16": _median(speedups),
                "max_forward_speedup_vs_bf16": max(speedups),
                "min_forward_speedup_vs_bf16": min(speedups),
                "median_quant_share": _median(quant_shares),
            }
        )
    return summaries


def _component_breakdown(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    wanted = {
        "dense_linear_bf16",
        "dense_linear_fp32",
        "activation_scale_plus_quant",
        "nvfp4_gemm_only",
        "nvfp4_forward_like",
    }
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("op") not in wanted:
            continue
        grouped[(row["shape_family"], row["op"])].append(_to_float(row["latency_mean_ms"]))
    output = []
    for (family, op), values in sorted(grouped.items()):
        output.append(
            {
                "shape_family": family,
                "op": op,
                "count": len(values),
                "median_ms": _median(values),
                "mean_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        )
    return output


def _plot_outputs(
    output_dir: Path,
    shape_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    _plot_scatter(
        output_dir / "speedup_vs_m.png",
        shape_rows,
        "m",
        "forward_speedup_vs_bf16",
        "M",
        "NVFP4 forward speedup vs dense bf16",
        log_x=True,
    )
    _plot_scatter(
        output_dir / "speedup_vs_arithmetic_intensity.png",
        shape_rows,
        "arithmetic_intensity_bf16",
        "forward_speedup_vs_bf16",
        "Arithmetic intensity, bf16 estimate",
        "NVFP4 forward speedup vs dense bf16",
        log_x=True,
    )
    _plot_components(output_dir / "quant_gemm_breakdown.png", component_rows)
    _plot_fixed_dimension_sweeps(output_dir / "fixed_dimension_sweep_speedups.png", shape_rows)
    _plot_fixed_dimension_breakdowns(output_dir / "fixed_dimension_sweep_breakdown.png", shape_rows)
    plt.close("all")


def _plot_scatter(
    path: Path,
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    log_x: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    families = sorted({str(row["shape_family"]) for row in rows})
    fig, ax = plt.subplots(figsize=(10, 6))
    for family in families:
        family_rows = [row for row in rows if row["shape_family"] == family]
        ax.scatter(
            [_to_float(row[x_key]) for row in family_rows],
            [_to_float(row[y_key]) for row in family_rows],
            label=family,
            s=28,
            alpha=0.8,
        )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def _plot_components(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    families = sorted({str(row["shape_family"]) for row in rows})
    ops = ["activation_scale_plus_quant", "nvfp4_gemm_only", "nvfp4_forward_like", "dense_linear_bf16"]
    by_key = {(row["shape_family"], row["op"]): _to_float(row["median_ms"]) for row in rows}
    width = 0.2
    x_positions = list(range(len(families)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for op_index, op in enumerate(ops):
        offset = (op_index - (len(ops) - 1) / 2) * width
        values = [by_key.get((family, op), math.nan) for family in families]
        ax.bar([x + offset for x in x_positions], values, width=width, label=op)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("Median latency (ms)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def _plot_fixed_dimension_sweeps(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for ax, (family, x_key, subtitle) in zip(axes.flat, _fixed_dimension_sweep_specs()):
        family_rows = sorted(
            [row for row in rows if row["shape_family"] == family],
            key=lambda row: _to_float(row[x_key]),
        )
        x_values = [_to_float(row[x_key]) for row in family_rows]
        bf16_speedups = [_to_float(row["forward_speedup_vs_bf16"]) for row in family_rows]
        fp32_speedups = [_to_float(row["forward_speedup_vs_fp32"]) for row in family_rows]
        ax.plot(x_values, bf16_speedups, marker="o", linewidth=1.8, label="vs dense bf16")
        ax.plot(x_values, fp32_speedups, marker="s", linewidth=1.8, label="vs dense fp32")
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{family}\n{subtitle}", fontsize=10)
        ax.set_xlabel(x_key.upper())
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("NVFP4 forward-like speedup")
    axes[1][0].set_ylabel("NVFP4 forward-like speedup")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False)
    fig.suptitle("Fixed-dimension sweep speedups", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)


def _plot_fixed_dimension_breakdowns(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=False)
    colors = {
        "quant": "#4c78a8",
        "gemm": "#f58518",
        "other": "#54a24b",
    }
    for ax, (family, x_key, subtitle) in zip(axes.flat, _fixed_dimension_sweep_specs()):
        family_rows = sorted(
            [row for row in rows if row["shape_family"] == family],
            key=lambda row: _to_float(row[x_key]),
        )
        x_labels = [_format_dimension_tick(row[x_key]) for row in family_rows]
        positions = list(range(len(family_rows)))
        quant = [_to_float(row["activation_scale_plus_quant_ms"]) for row in family_rows]
        gemm = [_to_float(row["nvfp4_gemm_only_ms"]) for row in family_rows]
        forward = [_to_float(row["nvfp4_forward_like_ms"]) for row in family_rows]
        other = [max(total - q - g, 0.0) for total, q, g in zip(forward, quant, gemm)]
        ax.bar(positions, quant, color=colors["quant"], label="activation scale+quant")
        ax.bar(positions, gemm, bottom=quant, color=colors["gemm"], label="NVFP4 GEMM-only")
        ax.bar(
            positions,
            other,
            bottom=[q + g for q, g in zip(quant, gemm)],
            color=colors["other"],
            label="other forward overhead",
        )
        ax.set_title(f"{family}\n{subtitle}", fontsize=10)
        ax.set_xlabel(x_key.upper())
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("NVFP4 forward-like latency (ms)")
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False)
    fig.suptitle("Fixed-dimension sweep NVFP4 forward-like breakdown", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)


def _fixed_dimension_sweep_specs() -> list[tuple[str, str, str]]:
    return [
        ("m_sweep_large_context", "m", "large: fixed n=4096, k=4096"),
        ("m_sweep_small_context", "m", "small: fixed n=512, k=512"),
        ("n_sweep_large_context", "n", "large: fixed m=4096, k=4096"),
        ("n_sweep_small_context", "n", "small: fixed m=512, k=512"),
        ("k_sweep_large_context", "k", "large: fixed m=4096, n=4096"),
        ("k_sweep_small_context", "k", "small: fixed m=512, n=512"),
    ]


def _format_dimension_tick(value: object) -> str:
    numeric = int(_to_float(value))
    if numeric >= 1024 and numeric % 1024 == 0:
        return f"{numeric // 1024}K"
    return str(numeric)


def _write_markdown(
    path: Path,
    input_path: Path,
    shape_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    raw_rows: list[dict[str, str]],
) -> None:
    ok_shapes = len(shape_rows)
    error_rows = [row for row in raw_rows if row.get("status") == "ERROR"]
    lines = [
        "# FlashInfer Custom Shape Benchmark Summary",
        "",
        f"Input CSV: `{input_path}`",
        f"OK shapes: {ok_shapes}",
        f"ERROR rows: {len(error_rows)}",
        "",
    ]
    if family_rows:
        lines.extend(
            [
                "## Shape Family Speedup",
                "",
                "| shape_family | shapes | accelerated | median speedup vs bf16 | median quant share |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in family_rows:
            lines.append(
                "| {shape_family} | {shapes} | {accelerated_vs_bf16} | {median_forward_speedup_vs_bf16:.3f}x | {median_quant_share:.1%} |".format(
                    **row
                )
            )
        lines.append("")
    if shape_rows:
        top = sorted(shape_rows, key=lambda row: _to_float(row["forward_speedup_vs_bf16"]), reverse=True)[:10]
        lines.extend(
            [
                "## Top Speedups",
                "",
                "| family | shape `(m,n,k)` | dense bf16 ms | NVFP4 forward ms | speedup |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in top:
            lines.append(
                "| {shape_family} | `{m},{n},{k}` | {dense_linear_bf16_ms:.3f} | {nvfp4_forward_like_ms:.3f} | {forward_speedup_vs_bf16:.3f}x |".format(
                    **row
                )
            )
        lines.extend(
            [
                "",
                "Generated files:",
                "",
                "- `summary.csv`",
                "- `speedup_by_shape_family.csv`",
                "- `component_breakdown.csv`",
                "- `speedup_vs_m.png`",
                "- `speedup_vs_arithmetic_intensity.png`",
                "- `quant_gemm_breakdown.png`",
                "- `fixed_dimension_sweep_speedups.png`",
                "- `fixed_dimension_sweep_breakdown.png`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_op(ops: dict[str, dict[str, str]], op: str) -> float | None:
    row = ops.get(op)
    if row is None:
        return None
    value = row.get("latency_mean_ms")
    if value in (None, ""):
        return None
    return _to_float(value)


def _to_float(value: object) -> float:
    return float(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


if __name__ == "__main__":
    main()
