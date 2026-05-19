#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

MODELS = [
    ("maxvit_tiny", "Tiny"),
    ("maxvit_small", "Small"),
    ("maxvit_base", "Base"),
    ("maxvit_large", "Large"),
    ("dinov3_vit7b16", "DINOv3"),
]

METHODS = {
    "unstructured": {
        "title": "NVFP4 + Unstructured Sparse",
        "baseline": "nvfp4_unstructured_sparse",
        "rescale": "nvfp4_4over6_unstructured_sparse",
        "real_ratio_checkpoint": "cutlass_nvfp4_runtime",
    },
    "structured": {
        "title": "NVFP4 + 4:8 Structured Sparse",
        "baseline": "nvfp4_semi_structured_sparse",
        "rescale": "nvfp4_4over6_semi_structured_sparse",
        "real_ratio_checkpoint": "cutlass_sparse_nvfp4_storage",
    },
}


@dataclass(frozen=True)
class SelectedRow:
    model_key: str
    model_label: str
    family: str
    method_label: str
    method: str
    selection: str
    seed: str
    top1: float
    top5: float
    dense_top1: float
    compression_ratio: float | None
    compression_ratio_source: str
    activation_quant: str
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot rescale accuracy/compression summary for PPT.")
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--file-ratio-csv", default="artifacts/results/checkpoint_file_compression_ratios.csv")
    parser.add_argument("--csv-output", default="artifacts/results/rescale_accuracy_compression_summary.csv")
    parser.add_argument("--png-output", default="artifacts/results/rescale_accuracy_compression_summary.png")
    parser.add_argument("--pdf-output", default="artifacts/results/rescale_accuracy_compression_summary.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    file_ratios = _load_file_ratios(Path(args.file_ratio_csv))
    rows = _collect_rows(results_dir, file_ratios)
    _write_csv(rows, Path(args.csv_output))
    _plot(rows, Path(args.png_output), Path(args.pdf_output))
    print(f"[plot] wrote {args.csv_output}")
    print(f"[plot] wrote {args.png_output}")
    print(f"[plot] wrote {args.pdf_output}")


def _collect_rows(
    results_dir: Path,
    file_ratios: dict[tuple[str, str], tuple[float, str]],
) -> list[SelectedRow]:
    rows: list[SelectedRow] = []
    for model_key, model_label in MODELS:
        dense = _select_dense(results_dir, model_key)
        if dense is None:
            continue
        dense_top1 = _percent(dense.get("top1", ""))
        for family, methods in METHODS.items():
            baseline = _select_seeded(
                rows=_method_rows(results_dir, model_key, methods["baseline"]),
                method=methods["baseline"],
                activation_quant="False",
                choose="worst",
            )
            rescale = _select_seeded(
                rows=_method_rows(results_dir, model_key, methods["rescale"]),
                method=methods["rescale"],
                activation_quant="False",
                choose="best",
            )
            if baseline is not None:
                rows.append(
                    _selected_row(
                        source=baseline,
                        dense_top1=dense_top1,
                        model_key=model_key,
                        model_label=model_label,
                        family=family,
                        method_label="Original NVFP4",
                        selection="worst seed",
                        file_ratios=file_ratios,
                    )
                )
            if rescale is not None:
                rows.append(
                    _selected_row(
                        source=rescale,
                        dense_top1=dense_top1,
                        model_key=model_key,
                        model_label=model_label,
                        family=family,
                        method_label="Rescale",
                        selection="best seed, act. off",
                        file_ratios=file_ratios,
                    )
                )
    return rows


def _selected_row(
    source: dict[str, str],
    dense_top1: float,
    model_key: str,
    model_label: str,
    family: str,
    method_label: str,
    selection: str,
    file_ratios: dict[tuple[str, str], tuple[float, str]],
) -> SelectedRow:
    compression_ratio, compression_ratio_source = _real_file_ratio_for_family(
        file_ratios=file_ratios,
        model_key=model_key,
        family=family,
    )
    return SelectedRow(
        model_key=model_key,
        model_label=model_label,
        family=family,
        method_label=method_label,
        method=source.get("method", ""),
        selection=selection,
        seed=source.get("seed", _seed_from_checkpoint(source.get("checkpoint_path", "")) or ""),
        top1=_percent(source.get("top1", "")),
        top5=_percent(source.get("top5", "")),
        dense_top1=dense_top1,
        compression_ratio=compression_ratio,
        compression_ratio_source=compression_ratio_source,
        activation_quant=source.get("activation_quant") or "False",
        checkpoint_path=source.get("checkpoint_path", ""),
    )


def _method_rows(results_dir: Path, model_key: str, method: str) -> list[dict[str, str]]:
    paths: list[Path]
    if model_key == "dinov3_vit7b16":
        if method.startswith("nvfp4_4over6_unstructured"):
            paths = [results_dir / "dinov3_vit7b16_4over6_unstructured_sparse" / "accuracy_seeded.csv"]
        elif method.startswith("nvfp4_4over6_semi"):
            paths = [results_dir / "dinov3_vit7b16_4over6_semi_structured_sparse" / "accuracy_seeded.csv"]
        else:
            paths = [results_dir / "dinov3_vit7b16_compressed" / "accuracy_seeded.csv"]
    else:
        if method.startswith("nvfp4_4over6"):
            paths = [results_dir / f"{model_key}_4over6" / "accuracy.csv"]
        else:
            paths = [results_dir / f"{model_key}_compressed" / "accuracy.csv"]
    found: list[dict[str, str]] = []
    for path in paths:
        for row in _read_csv(path):
            if row.get("method") == method:
                found.append(row)
    return found


def _select_seeded(
    rows: list[dict[str, str]],
    method: str,
    activation_quant: str,
    choose: str,
) -> dict[str, str] | None:
    candidates = [
        row for row in rows
        if row.get("method") == method
        and (row.get("activation_quant") or "False") == activation_quant
        and _seed(row) != ""
    ]
    if not candidates:
        return None
    key = lambda row: _percent(row.get("top1", ""))
    return min(candidates, key=key) if choose == "worst" else max(candidates, key=key)


def _select_dense(results_dir: Path, model_key: str) -> dict[str, str] | None:
    path = results_dir / f"{model_key}_dense" / "accuracy.csv"
    latest: dict[str, str] | None = None
    for row in _read_csv(path):
        if row.get("method") != "dense":
            continue
        latest = row
    return latest


def _load_file_ratios(path: Path) -> dict[tuple[str, str], tuple[float, str]]:
    ratios: dict[tuple[str, str], tuple[float, str]] = {}
    for row in _read_csv(path):
        model_key = row.get("model_key", "")
        checkpoint = row.get("checkpoint", "")
        if not model_key or not checkpoint:
            continue
        try:
            ratio = float(row.get("file_size_ratio", ""))
        except ValueError:
            continue
        source = row.get("path", "") or checkpoint
        ratios[(model_key, checkpoint)] = (ratio, source)
    return ratios


def _real_file_ratio_for_family(
    file_ratios: dict[tuple[str, str], tuple[float, str]],
    model_key: str,
    family: str,
) -> tuple[float | None, str]:
    checkpoint = METHODS[family]["real_ratio_checkpoint"]
    ratio = file_ratios.get((model_key, checkpoint))
    if ratio is None:
        return None, checkpoint
    return ratio


def _write_csv(rows: list[SelectedRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "family",
        "method_label",
        "method",
        "selection",
        "seed",
        "top1",
        "top5",
        "dense_top1",
        "top1_drop_vs_dense",
        "compression_ratio",
        "compression_ratio_source",
        "activation_quant",
        "checkpoint_path",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model_label,
                    "family": row.family,
                    "method_label": row.method_label,
                    "method": row.method,
                    "selection": row.selection,
                    "seed": row.seed,
                    "top1": f"{row.top1:.3f}",
                    "top5": f"{row.top5:.3f}",
                    "dense_top1": f"{row.dense_top1:.3f}",
                    "top1_drop_vs_dense": f"{row.dense_top1 - row.top1:.3f}",
                    "compression_ratio": "" if row.compression_ratio is None else f"{row.compression_ratio:.3f}",
                    "compression_ratio_source": row.compression_ratio_source,
                    "activation_quant": row.activation_quant,
                    "checkpoint_path": row.checkpoint_path,
                }
            )


def _plot(rows: list[SelectedRow], png_output: Path, pdf_output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#2b2b2b",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.9), sharey=False)
    colors = {"Original NVFP4": "#6f7785", "Rescale": "#0f8b8d"}
    hatches = {"Original NVFP4": "", "Rescale": "///"}

    for ax, family in zip(axes, ("unstructured", "structured"), strict=True):
        family_rows = [row for row in rows if row.family == family]
        title = METHODS[family]["title"]
        _plot_family(ax, family_rows, title, colors, hatches)

    handles = [
        Patch(facecolor=colors["Original NVFP4"], edgecolor="#222222", label="Original NVFP4"),
        Patch(facecolor=colors["Rescale"], edgecolor="#222222", hatch=hatches["Rescale"], label="Rescale"),
        Line2D([0], [0], color="#1f1f1f", linestyle="--", linewidth=1.1, label="Dense baseline (FP32)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=11.5,
        handlelength=2.0,
        columnspacing=2.2,
    )
    fig.suptitle("Accuracy vs. Compression Ratio Across Vision Models", fontsize=15, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.01,
        "Bar height: ImageNet Top-1 accuracy.  CR labels use real file-size ratios from non-Rescale packed checkpoints; Rescale reuses the matching Original ratio.",
        ha="center",
        fontsize=9,
        color="#4d4d4d",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    png_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)


def _plot_family(
    ax,
    rows: list[SelectedRow],
    title: str,
    colors: dict[str, str],
    hatches: dict[str, str],
) -> None:
    labels = [label for _key, label in MODELS]
    x = list(range(len(labels)))
    width = 0.34
    by_key = {(row.model_key, row.method_label): row for row in rows}
    effective_cr_by_key = {}
    for model_key, _label in MODELS:
        original = by_key.get((model_key, "Original NVFP4"))
        rescale = by_key.get((model_key, "Rescale"))
        effective_cr_by_key[model_key] = (
            original.compression_ratio
            if original is not None and original.compression_ratio is not None
            else (rescale.compression_ratio if rescale is not None else None)
        )

    original_values = [by_key.get((key, "Original NVFP4")).top1 if by_key.get((key, "Original NVFP4")) else math.nan for key, _ in MODELS]
    rescale_values = [by_key.get((key, "Rescale")).top1 if by_key.get((key, "Rescale")) else math.nan for key, _ in MODELS]

    ax.bar(
        [i - width / 2 for i in x],
        original_values,
        width,
        color=colors["Original NVFP4"],
        edgecolor="#222222",
        linewidth=0.7,
    )
    ax.bar(
        [i + width / 2 for i in x],
        rescale_values,
        width,
        color=colors["Rescale"],
        edgecolor="#222222",
        linewidth=0.7,
        hatch=hatches["Rescale"],
    )

    dense_values = [
        max((row.dense_top1 for row in rows if row.model_key == key), default=math.nan)
        for key, _ in MODELS
    ]
    ax.plot(x, dense_values, color="#1f1f1f", linestyle="--", linewidth=1.1, alpha=0.55, zorder=3)

    valid_values = [v for v in original_values + rescale_values + dense_values if not math.isnan(v)]
    ymin = max(0.0, min(valid_values) - 4.0)
    ymax = min(100.0, max(valid_values) + 5.0)
    if ymax - ymin < 8.0:
        ymin = max(0.0, ymax - 8.0)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx, (model_key, _label) in enumerate(MODELS):
        original = by_key.get((model_key, "Original NVFP4"))
        rescale = by_key.get((model_key, "Rescale"))
        if original is None or rescale is None:
            continue
        ax.text(idx - width / 2, original.top1 + 0.18, f"{original.top1:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, rescale.top1 + 0.18, f"{rescale.top1:.2f}", ha="center", va="bottom", fontsize=8)
        cr = effective_cr_by_key.get(model_key)
        cr_text = "CR n/a" if cr is None else f"CR {cr:.2f}x"
        y = max(original.top1, rescale.top1) + 2.2
        ax.text(
            idx,
            y,
            cr_text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#0f6b38",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=5,
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _seed(row: dict[str, str]) -> str:
    return row.get("seed") or _seed_from_checkpoint(row.get("checkpoint_path", "")) or ""


def _seed_from_checkpoint(checkpoint_path: str) -> str | None:
    match = re.search(r"_seed(\d+)/model\.pt$", checkpoint_path)
    return match.group(1) if match else None


def _percent(value: str) -> float:
    return float(value) * 100.0


if __name__ == "__main__":
    main()
