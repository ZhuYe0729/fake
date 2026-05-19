#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_accuracy_results import _compression_ratio_for_row


METHOD_ORDER = [
    "dense",
    "nvfp4",
    "int4",
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
    "int4": "INT4",
    "unstructured_sparse": "Unstructured",
    "semi_structured_sparse": "2:4 Sparse",
    "nvfp4_unstructured_sparse": "NVFP4+Unstruct",
    "nvfp4_semi_structured_sparse": "NVFP4+2:4",
    "nvfp4_4over6_unstructured_sparse": "4/6+Unstruct",
    "nvfp4_4over6_semi_structured_sparse": "4/6+2:4",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    dense_accuracy_csv: Path
    dense_speed_csv: Path
    compressed_accuracy_csv: Path
    cutlass_nvfp4_accuracy_csv: Path
    cutlass_nvfp4_speed_csv: Path
    cutlass_sparse_accuracy_csv: Path
    cutlass_sparse_speed_csv: Path
    cutlass_sparse_bf16_accuracy_csv: Path
    cutlass_sparse_bf16_speed_csv: Path
    dense_speed_batch: int | None = None
    cutlass_speed_batch: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy, compression ratio, and speedup summary.")
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--output", default="artifacts/results/accuracy_compression_speed_summary.png")
    parser.add_argument("--csv-output", default="artifacts/results/accuracy_compression_speed_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    specs = _model_specs(results_dir)
    table = _collect_table(specs)
    _write_table_csv(table, Path(args.csv_output))
    _plot_summary(specs, table, Path(args.output))
    print(f"[plot] wrote {args.csv_output}")
    print(f"[plot] wrote {args.output}")


def _model_specs(results_dir: Path) -> list[ModelSpec]:
    specs = [
        _maxvit_spec(results_dir, "tiny", "MaxViT Tiny"),
        _maxvit_spec(results_dir, "small", "MaxViT Small"),
        _maxvit_spec(results_dir, "base", "MaxViT Base"),
        _maxvit_spec(results_dir, "large", "MaxViT Large"),
    ]
    specs.append(
        ModelSpec(
            key="dinov3_vit7b16",
            title="DINOv3 ViT-7B/16",
            dense_accuracy_csv=results_dir / "dinov3_vit7b16_dense" / "accuracy.csv",
            dense_speed_csv=results_dir / "dinov3_vit7b16_dense" / "speed.csv",
            compressed_accuracy_csv=results_dir / "dinov3_vit7b16_compressed" / "accuracy.csv",
            cutlass_nvfp4_accuracy_csv=results_dir / "dinov3_vit7b16_cutlass_nvfp4" / "accuracy.csv",
            cutlass_nvfp4_speed_csv=results_dir / "dinov3_vit7b16_cutlass_nvfp4" / "speed.csv",
            cutlass_sparse_accuracy_csv=results_dir / "dinov3_vit7b16_cutlass_sparse_nvfp4" / "accuracy_storage.csv",
            cutlass_sparse_speed_csv=results_dir / "dinov3_vit7b16_cutlass_sparse_nvfp4" / "speed_storage.csv",
            cutlass_sparse_bf16_accuracy_csv=results_dir / "dinov3_vit7b16_cutlass_sparse_bf16" / "accuracy.csv",
            cutlass_sparse_bf16_speed_csv=results_dir / "dinov3_vit7b16_cutlass_sparse_bf16" / "speed.csv",
            dense_speed_batch=8,
            cutlass_speed_batch=8,
        )
    )
    return specs


def _maxvit_spec(results_dir: Path, variant: str, title: str) -> ModelSpec:
    compressed_dir = results_dir / f"maxvit_{variant}_compressed"
    if variant == "tiny" and not compressed_dir.exists():
        compressed_dir = results_dir / "maxvit_compressed"
    return ModelSpec(
        key=f"maxvit_{variant}",
        title=title,
        dense_accuracy_csv=results_dir / f"maxvit_{variant}_dense" / "accuracy.csv",
        dense_speed_csv=results_dir / f"maxvit_{variant}_dense" / "speed.csv",
        compressed_accuracy_csv=compressed_dir / "accuracy.csv",
        cutlass_nvfp4_accuracy_csv=results_dir / f"maxvit_{variant}_cutlass_nvfp4" / "accuracy.csv",
        cutlass_nvfp4_speed_csv=results_dir / f"maxvit_{variant}_cutlass_nvfp4" / "speed.csv",
        cutlass_sparse_accuracy_csv=results_dir / f"maxvit_{variant}_cutlass_sparse_nvfp4" / "accuracy.csv",
        cutlass_sparse_speed_csv=results_dir / f"maxvit_{variant}_cutlass_sparse_nvfp4" / "speed.csv",
        cutlass_sparse_bf16_accuracy_csv=results_dir / f"maxvit_{variant}_cutlass_sparse_bf16" / "accuracy.csv",
        cutlass_sparse_bf16_speed_csv=results_dir / f"maxvit_{variant}_cutlass_sparse_bf16" / "speed.csv",
    )


def _collect_table(specs: list[ModelSpec]) -> dict[str, dict[str, dict[str, float | None]]]:
    table: dict[str, dict[str, dict[str, float | None]]] = {}
    for spec in specs:
        dense_accuracy = _latest_method_row(spec.dense_accuracy_csv, "dense")
        dense_speed = _latest_speed_row(spec.dense_speed_csv, spec.dense_speed_batch)
        dense_ips = _float_or_none(dense_speed.get("images_per_sec") if dense_speed else None)
        rows: dict[str, dict[str, float | None]] = {}

        if dense_accuracy:
            rows["dense"] = {
                "top1": _percent(dense_accuracy.get("top1")),
                "compression_ratio": 1.0,
                "speedup": 1.0,
            }

        compressed_rows = {row.get("method", ""): row for row in _latest_rows_by_method(spec.compressed_accuracy_csv)}
        for method in ("unstructured_sparse", "semi_structured_sparse", "nvfp4_unstructured_sparse"):
            row = compressed_rows.get(method)
            if row:
                rows[method] = {
                    "top1": _percent(row.get("top1")),
                    "compression_ratio": _safe_compression_ratio(row),
                    "speedup": None,
                }
        _apply_sparse_bf16_result(spec, rows, dense_ips)
        if spec.key == "dinov3_vit7b16":
            for method, accuracy_csv, speed_csv in _dinov3_four_over_six_paths(spec.dense_accuracy_csv.parent.parent):
                row = _latest_method_row(accuracy_csv, method)
                speed = _latest_speed_row(speed_csv, spec.dense_speed_batch)
                if row:
                    ips = _float_or_none(speed.get("images_per_sec") if speed else None)
                    rows[method] = {
                        "top1": _percent(row.get("top1")),
                        "compression_ratio": _safe_compression_ratio(row),
                        "speedup": _speedup(ips, dense_ips),
                    }

        nvfp4_accuracy = _latest_method_row(spec.cutlass_nvfp4_accuracy_csv, "nvfp4_cutlass")
        nvfp4_speed = _latest_speed_row(spec.cutlass_nvfp4_speed_csv, spec.cutlass_speed_batch)
        if nvfp4_accuracy:
            nvfp4_accuracy = _with_default_checkpoint_path(spec, nvfp4_accuracy, "runtime_checkpoint_path")
            nvfp4_ips = _float_or_none(nvfp4_speed.get("images_per_sec") if nvfp4_speed else None)
            rows["nvfp4"] = {
                "top1": _percent(nvfp4_accuracy.get("top1")),
                "compression_ratio": _actual_checkpoint_ratio(nvfp4_accuracy, "runtime_checkpoint_path"),
                "speedup": _speedup(nvfp4_ips, dense_ips),
            }

        sparse_accuracy = _latest_method_row(spec.cutlass_sparse_accuracy_csv, "sparse_nvfp4_cutlass")
        sparse_speed = _latest_speed_row(spec.cutlass_sparse_speed_csv, spec.cutlass_speed_batch)
        if sparse_accuracy:
            sparse_accuracy = _with_default_checkpoint_path(spec, sparse_accuracy, "storage_checkpoint_path")
            sparse_ips = _float_or_none(sparse_speed.get("images_per_sec") if sparse_speed else None)
            rows["nvfp4_semi_structured_sparse"] = {
                "top1": _percent(sparse_accuracy.get("top1")),
                "compression_ratio": _actual_checkpoint_ratio(sparse_accuracy, "storage_checkpoint_path"),
                "speedup": _speedup(sparse_ips, dense_ips),
            }

        table[spec.key] = rows
    return table


def _apply_sparse_bf16_result(
    spec: ModelSpec,
    rows: dict[str, dict[str, float | None]],
    dense_ips: float | None,
) -> None:
    accuracy = _latest_method_row(spec.cutlass_sparse_bf16_accuracy_csv, "sparse_bf16_cutlass")
    speed = _latest_speed_row(spec.cutlass_sparse_bf16_speed_csv, spec.cutlass_speed_batch)
    if accuracy is None and speed is None:
        return

    existing = rows.get("semi_structured_sparse", {})
    sparse_ips = _float_or_none(speed.get("images_per_sec") if speed else None)
    rows["semi_structured_sparse"] = {
        "top1": _percent(accuracy.get("top1")) if accuracy else existing.get("top1"),
        "compression_ratio": _sparse_bf16_compression_ratio(accuracy) if accuracy else existing.get("compression_ratio"),
        "speedup": _speedup(sparse_ips, dense_ips),
    }


def _sparse_bf16_compression_ratio(row: dict[str, str] | None) -> float | None:
    if row is None:
        return None
    for checkpoint_key in ("storage_checkpoint_path", "checkpoint_path", "runtime_checkpoint_path"):
        ratio = _actual_checkpoint_ratio(row, checkpoint_key)
        if ratio is not None:
            return ratio
    return _safe_compression_ratio(row)


def _with_default_checkpoint_path(spec: ModelSpec, row: dict[str, str], key: str) -> dict[str, str]:
    if row.get(key):
        return row
    row = dict(row)
    if key == "runtime_checkpoint_path":
        if spec.key == "dinov3_vit7b16":
            row[key] = "artifacts/checkpoints/dinov3_vit7b16/cutlass_nvfp4_runtime/model.pt"
        elif spec.key.startswith("maxvit_"):
            variant = spec.key.removeprefix("maxvit_")
            row[key] = f"artifacts/checkpoints/maxvit_{variant}/cutlass_nvfp4_runtime/model.pt"
    elif key == "storage_checkpoint_path":
        if spec.key == "dinov3_vit7b16":
            row[key] = "artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/model.pt"
        elif spec.key.startswith("maxvit_"):
            variant = spec.key.removeprefix("maxvit_")
            row[key] = f"artifacts/checkpoints/maxvit_{variant}/cutlass_sparse_nvfp4_storage/model.pt"
    return row


def _latest_rows_by_method(path: Path) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        method = row.get("method", "")
        if method and _timestamp(row) >= _timestamp(latest.get(method, {})):
            latest[method] = row
    return list(latest.values())


def _latest_method_row(path: Path, method: str) -> dict[str, str] | None:
    latest: dict[str, str] | None = None
    for row in _read_csv(path):
        if row.get("method") != method:
            continue
        if latest is None or _timestamp(row) >= _timestamp(latest):
            latest = row
    return latest


def _latest_speed_row(path: Path, batch_size: int | None) -> dict[str, str] | None:
    latest: dict[str, str] | None = None
    for row in _read_csv(path):
        if batch_size is not None and row.get("batch_size") != str(batch_size):
            continue
        if latest is None or _timestamp(row) >= _timestamp(latest):
            latest = row
    return latest


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _dinov3_four_over_six_paths(results_dir: Path) -> list[tuple[str, Path, Path]]:
    return [
        (
            "nvfp4_4over6_unstructured_sparse",
            results_dir / "dinov3_vit7b16_4over6_unstructured_sparse" / "accuracy.csv",
            results_dir / "dinov3_vit7b16_4over6_unstructured_sparse" / "speed.csv",
        ),
        (
            "nvfp4_4over6_semi_structured_sparse",
            results_dir / "dinov3_vit7b16_4over6_semi_structured_sparse" / "accuracy.csv",
            results_dir / "dinov3_vit7b16_4over6_semi_structured_sparse" / "speed.csv",
        ),
    ]


def _timestamp(row: dict[str, str]) -> datetime:
    try:
        return datetime.fromisoformat(row.get("timestamp", ""))
    except ValueError:
        return datetime.min


def _percent(value: str | None) -> float | None:
    parsed = _float_or_none(value)
    return None if parsed is None else parsed * 100.0


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_compression_ratio(row: dict[str, str]) -> float | None:
    try:
        value = _compression_ratio_for_row(row)
    except Exception:
        return None
    return None if math.isnan(value) else value


def _actual_checkpoint_ratio(row: dict[str, str], checkpoint_key: str) -> float | None:
    dense_bytes = _dense_source_bytes(row)
    checkpoint_path = _resolve_path(row.get(checkpoint_key, ""))
    if checkpoint_path is None or not checkpoint_path.exists() or dense_bytes <= 0:
        return _safe_compression_ratio(row)
    return dense_bytes / checkpoint_path.stat().st_size


def _dense_source_bytes(row: dict[str, str]) -> int:
    total = 0
    model_path = _resolve_path(row.get("model_path", ""))
    backbone_path = _resolve_path(row.get("backbone_path", ""))
    head_path = _resolve_path(row.get("head_path", ""))
    if model_path is not None:
        total += _path_bytes(model_path)
    if backbone_path is not None:
        total += _path_bytes(backbone_path)
    if head_path is not None:
        total += _path_bytes(head_path)
    return total


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.glob("*.safetensors"))
    return 0


def _resolve_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _speedup(images_per_sec: float | None, dense_images_per_sec: float | None) -> float | None:
    if images_per_sec is None or dense_images_per_sec is None or dense_images_per_sec <= 0:
        return None
    return images_per_sec / dense_images_per_sec


def _write_table_csv(table: dict[str, dict[str, dict[str, float | None]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "method", "top1_percent", "compression_ratio", "speedup"],
        )
        writer.writeheader()
        for model, rows in table.items():
            for method in METHOD_ORDER:
                values = rows.get(method)
                if values is None:
                    continue
                writer.writerow(
                    {
                        "model": model,
                        "method": method,
                        "top1_percent": _format_float(values.get("top1")),
                        "compression_ratio": _format_float(values.get("compression_ratio")),
                        "speedup": _format_float(values.get("speedup")),
                    }
                )


def _plot_summary(
    specs: list[ModelSpec],
    table: dict[str, dict[str, dict[str, float | None]]],
    output_path: Path,
) -> None:
    matrix = [
        [table.get(spec.key, {}).get(method, {}).get("top1") for method in METHOD_ORDER]
        for spec in specs
    ]
    fig, ax = plt.subplots(figsize=(15.5, 6.6), constrained_layout=True)
    image = ax.imshow(
        [[value if value is not None else float("nan") for value in row] for row in matrix],
        cmap="viridis",
        aspect="auto",
        vmin=0,
        vmax=100,
    )
    ax.set_title("Top-1 Accuracy, Compression Ratio, and Speedup Summary", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_LABELS[method] for method in METHOD_ORDER], rotation=24, ha="right")
    ax.set_yticks(range(len(specs)), [spec.title for spec in specs])

    for row_idx, spec in enumerate(specs):
        for col_idx, method in enumerate(METHOD_ORDER):
            values = table.get(spec.key, {}).get(method)
            text = _cell_text(values)
            top1 = values.get("top1") if values else None
            color = "white" if top1 is not None and top1 < 65.0 else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=8.2, linespacing=1.15)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Top-1 Accuracy (%)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _cell_text(values: dict[str, float | None] | None) -> str:
    if values is None:
        return "NA"
    top1 = values.get("top1")
    cr = values.get("compression_ratio")
    speedup = values.get("speedup")
    top1_text = "Acc NA" if top1 is None else f"Acc {top1:.2f}"
    cr_text = "CR NA" if cr is None else f"CR {cr:.2f}x"
    speed_text = "Speed NA" if speedup is None else f"Speed {speedup:.2f}x"
    return f"{top1_text}\n{cr_text}\n{speed_text}"


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
