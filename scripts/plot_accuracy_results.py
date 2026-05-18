#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = [
    "dense",
    "nvfp4",
    "unstructured_sparse",
    "semi_structured_sparse",
    "nvfp4_unstructured_sparse",
    "nvfp4_semi_structured_sparse",
    "nvfp4_4over6_unstructured_sparse",
    "nvfp4_4over6_semi_structured_sparse",
]
METHOD_LABELS = {
    "dense": "Dense",
    "nvfp4": "NVFP4",
    "unstructured_sparse": "Unstructured",
    "semi_structured_sparse": "2:4 Sparse",
    "nvfp4_unstructured_sparse": "NVFP4+Unstruct",
    "nvfp4_semi_structured_sparse": "NVFP4+2:4",
    "nvfp4_4over6_unstructured_sparse": "4/6+Unstruct",
    "nvfp4_4over6_semi_structured_sparse": "4/6+2:4",
}
METHOD_COLORS = {
    "dense": "#2f4858",
    "nvfp4": "#377eb8",
    "unstructured_sparse": "#4daf4a",
    "semi_structured_sparse": "#984ea3",
    "nvfp4_unstructured_sparse": "#ff7f00",
    "nvfp4_semi_structured_sparse": "#e41a1c",
    "nvfp4_4over6_unstructured_sparse": "#a65628",
    "nvfp4_4over6_semi_structured_sparse": "#f781bf",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    dense_csv: Path
    compressed_csv: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dense and compressed accuracy results.")
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--per-model-name", default="accuracy_comparison.png")
    parser.add_argument("--summary-name", default="accuracy_summary.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    specs = _model_specs(results_dir)
    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for spec in specs:
        rows = _collect_model_rows(spec)
        if not rows:
            print(f"[plot] skip {spec.key}: no accuracy rows")
            continue
        output_path = spec.output_dir / args.per_model_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_model(spec, rows, output_path)
        all_results[spec.key] = rows
        print(f"[plot] wrote {output_path}")

    summary_path = results_dir / args.summary_name
    _plot_summary(specs, all_results, summary_path)
    print(f"[plot] wrote {summary_path}")


def _model_specs(results_dir: Path) -> list[ModelSpec]:
    return [
        _maxvit_spec(results_dir, "tiny", "MaxViT Tiny"),
        _maxvit_spec(results_dir, "small", "MaxViT Small"),
        _maxvit_spec(results_dir, "base", "MaxViT Base"),
        _maxvit_spec(results_dir, "large", "MaxViT Large"),
        ModelSpec(
            key="dinov3_vit7b16",
            title="DINOv3 ViT-7B/16",
            dense_csv=results_dir / "dinov3_vit7b16_dense" / "accuracy.csv",
            compressed_csv=results_dir / "dinov3_vit7b16_compressed" / "accuracy.csv",
            output_dir=results_dir / "dinov3_vit7b16_compressed",
        ),
    ]


def _maxvit_spec(results_dir: Path, variant: str, title: str) -> ModelSpec:
    dense_dir = results_dir / f"maxvit_{variant}_dense"
    compressed_dir = results_dir / f"maxvit_{variant}_compressed"
    if variant == "tiny" and not dense_dir.exists():
        dense_dir = results_dir / "maxvit_dense"
    if variant == "tiny" and not compressed_dir.exists():
        compressed_dir = results_dir / "maxvit_compressed"
    return ModelSpec(
        key=f"maxvit_{variant}",
        title=title,
        dense_csv=dense_dir / "accuracy.csv",
        compressed_csv=compressed_dir / "accuracy.csv",
        output_dir=compressed_dir,
    )


def _collect_model_rows(spec: ModelSpec) -> dict[str, dict[str, float]]:
    method_rows: dict[str, dict[str, str]] = {}
    for row in _read_csv(spec.dense_csv):
        if row.get("method") == "dense":
            method_rows["dense"] = _latest(method_rows.get("dense"), row)
    for row in _read_csv(spec.compressed_csv):
        method = row.get("method", "")
        if method in METHOD_ORDER and method != "dense":
            method_rows[method] = _latest(method_rows.get(method), row)
    if spec.key == "dinov3_vit7b16":
        for method, path in _dinov3_four_over_six_csvs(spec.dense_csv.parent.parent).items():
            for row in _read_csv(path):
                if row.get("method") == method:
                    method_rows[method] = _latest(method_rows.get(method), row)

    results: dict[str, dict[str, float]] = {}
    for method in METHOD_ORDER:
        row = method_rows.get(method)
        if row is None:
            continue
        results[method] = {
            "top1": _as_percent(row.get("top1", "")),
            "top5": _as_percent(row.get("top5", "")),
            "compression_ratio": _compression_ratio_for_row(row),
        }
    return results


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _dinov3_four_over_six_csvs(results_dir: Path) -> dict[str, Path]:
    return {
        "nvfp4_4over6_unstructured_sparse": results_dir
        / "dinov3_vit7b16_4over6_unstructured_sparse"
        / "accuracy.csv",
        "nvfp4_4over6_semi_structured_sparse": results_dir
        / "dinov3_vit7b16_4over6_semi_structured_sparse"
        / "accuracy.csv",
    }


def _latest(current: dict[str, str] | None, candidate: dict[str, str]) -> dict[str, str]:
    if current is None:
        return candidate
    if _timestamp(candidate) >= _timestamp(current):
        return candidate
    return current


def _timestamp(row: dict[str, str]) -> datetime:
    value = row.get("timestamp", "")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def _as_percent(value: str) -> float:
    return float(value) * 100.0


def _compression_ratio_for_row(row: dict[str, str]) -> float:
    method = row.get("method", "")
    if method == "dense":
        return 1.0

    checkpoint_path = _resolve_path(row.get("checkpoint_path", ""))
    metadata_path = checkpoint_path.parent / "metadata.json" if checkpoint_path is not None else None
    if metadata_path is None or not metadata_path.exists():
        return float("nan")

    with metadata_path.open("r") as f:
        metadata = json.load(f)

    total_params = _total_params_for_row(row)
    selected_weights = _selected_weight_count(metadata)
    if total_params <= 0 or selected_weights <= 0:
        return float("nan")

    zeros = _zero_weight_count(metadata)
    scale_count = _scale_count(metadata)
    uncompressed_weights = total_params - selected_weights

    dense_bits = total_params * 32.0
    compressed_bits = uncompressed_weights * 32.0

    if _has_quant(metadata):
        scale_bits = _scale_bits(metadata)
        stored_weights = selected_weights - zeros if _has_prune(metadata) else selected_weights
        compressed_bits += stored_weights * 4.0 + scale_count * scale_bits
    elif _has_prune(metadata):
        compressed_bits += (selected_weights - zeros) * 32.0
    else:
        compressed_bits += selected_weights * 32.0

    if compressed_bits <= 0:
        return float("nan")
    return dense_bits / compressed_bits


def _resolve_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _total_params_for_row(row: dict[str, str]) -> int:
    model_path = _resolve_path(row.get("model_path") or row.get("backbone_path", ""))
    if model_path is None:
        return 0

    model = row.get("model", "")
    if "dinov3" in model:
        return _safetensors_dir_param_count(model_path) + _dinov3_head_param_count(row)

    return _safetensors_dir_param_count(model_path)


def _safetensors_dir_param_count(model_dir: Path) -> int:
    paths = sorted(model_dir.glob("*.safetensors"))
    return sum(_safetensors_param_count(path) for path in paths)


def _safetensors_param_count(path: Path) -> int:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    total = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape = info.get("shape", [])
        total += math.prod(int(dim) for dim in shape)
    return total


def _dinov3_head_param_count(row: dict[str, str]) -> int:
    hidden_size = int(row.get("hidden_size") or 4096)
    num_classes = 1000
    return num_classes * (2 * hidden_size) + num_classes


def _selected_weight_count(metadata: dict[str, Any]) -> int:
    total = 0
    for record in metadata.get("modules", []):
        prune = record.get("prune")
        quant = record.get("quant")
        if isinstance(prune, dict) and prune.get("status") == "ok":
            total += int(prune.get("numel", 0))
        elif isinstance(quant, dict) and quant.get("status") == "ok":
            total += int(quant.get("num_groups", 0)) * int(quant.get("group_size", metadata.get("nvfp4_group_size", 16)))
    return total


def _zero_weight_count(metadata: dict[str, Any]) -> int:
    total = 0
    for record in metadata.get("modules", []):
        prune = record.get("prune")
        if isinstance(prune, dict) and prune.get("status") == "ok":
            total += int(prune.get("zeros", 0))
    return total


def _scale_count(metadata: dict[str, Any]) -> int:
    total = 0
    for record in metadata.get("modules", []):
        quant = record.get("quant")
        if isinstance(quant, dict) and quant.get("status") == "ok":
            total += int(quant.get("num_groups", 0))
    return total


def _scale_bits(metadata: dict[str, Any]) -> float:
    precision = metadata.get("nvfp4_scale_precision", "fp16")
    if precision in {"fp16", "bf16"}:
        return 16.0
    if precision == "fp32":
        return 32.0
    return 16.0


def _has_prune(metadata: dict[str, Any]) -> bool:
    return any(isinstance(record.get("prune"), dict) for record in metadata.get("modules", []))


def _has_quant(metadata: dict[str, Any]) -> bool:
    return any(isinstance(record.get("quant"), dict) for record in metadata.get("modules", []))


def _plot_model(spec: ModelSpec, rows: dict[str, dict[str, float]], output_path: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in rows]
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    for ax, metric, ylabel in (
        (axes[0], "top1", "Top-1 Accuracy (%)"),
        (axes[1], "top5", "Top-5 Accuracy (%)"),
    ):
        values = [rows[method][metric] for method in methods]
        bars = ax.bar(labels, values, color=colors, edgecolor="#202020", linewidth=0.7)
        ax.set_title(ylabel.replace(" Accuracy (%)", ""))
        ax.set_ylabel(ylabel)
        ax.set_ylim(_metric_ylim(values))
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=24)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.25,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(f"{spec.title}: Dense vs Compressed Accuracy", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _metric_ylim(values: list[float]) -> tuple[float, float]:
    floor = max(0.0, min(values) - 5.0)
    ceil = min(100.0, max(values) + 3.0)
    if ceil - floor < 8.0:
        floor = max(0.0, ceil - 8.0)
    return floor, ceil


def _plot_summary(
    specs: list[ModelSpec],
    all_results: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    specs = [spec for spec in specs if spec.key in all_results]
    matrix = [
        [all_results[spec.key].get(method, {}).get("top1") for method in METHOD_ORDER]
        for spec in specs
    ]

    fig, ax = plt.subplots(figsize=(13.5, 5.8), constrained_layout=True)
    image = ax.imshow(
        [[value if value is not None else float("nan") for value in row] for row in matrix],
        cmap="viridis",
        aspect="auto",
        vmin=0,
        vmax=100,
    )
    ax.set_title("Top-1 Accuracy and Estimated Compression Ratio Summary", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_LABELS[method] for method in METHOD_ORDER], rotation=24, ha="right")
    ax.set_yticks(range(len(specs)), [spec.title for spec in specs])

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            method = METHOD_ORDER[col_idx]
            ratio = all_results[specs[row_idx].key].get(method, {}).get("compression_ratio")
            ratio_text = "CR NA" if ratio is None or math.isnan(ratio) else f"CR {ratio:.2f}x"
            text = "NA" if value is None else f"{value:.2f}\n{ratio_text}"
            color = "white" if value is not None and value < 65.0 else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=9, linespacing=1.15)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Top-1 Accuracy (%)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
