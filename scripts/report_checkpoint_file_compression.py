#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSource:
    key: str
    label: str
    dense_paths: tuple[Path, ...]
    checkpoint_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report real file-size compression ratios for checkpoint files.")
    parser.add_argument("--checkpoint-root", default="artifacts/checkpoints")
    parser.add_argument("--csv-output", default="artifacts/results/checkpoint_file_compression_ratios.csv")
    parser.add_argument("--md-output", default="artifacts/results/checkpoint_file_compression_ratios.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    rows = _collect_rows(_model_sources(checkpoint_root))
    _write_csv(rows, Path(args.csv_output))
    _write_md(rows, Path(args.md_output))
    print(f"[report] wrote {args.csv_output}")
    print(f"[report] wrote {args.md_output}")


def _model_sources(checkpoint_root: Path) -> list[ModelSource]:
    timm_root = Path("/data/home/scxj523/run/wja/data/models/timm")
    facebook_root = Path("/data/home/scxj523/run/wja/data/models/facebook")
    return [
        ModelSource(
            "maxvit_tiny",
            "MaxViT Tiny",
            (timm_root / "maxvit_tiny_tf_224.in1k" / "model.safetensors",),
            checkpoint_root / "maxvit_tiny",
        ),
        ModelSource(
            "maxvit_small",
            "MaxViT Small",
            (timm_root / "maxvit_small_tf_224.in1k" / "model.safetensors",),
            checkpoint_root / "maxvit_small",
        ),
        ModelSource(
            "maxvit_base",
            "MaxViT Base",
            (timm_root / "maxvit_base_tf_224.in1k" / "model.safetensors",),
            checkpoint_root / "maxvit_base",
        ),
        ModelSource(
            "maxvit_large",
            "MaxViT Large",
            (timm_root / "maxvit_large_tf_512.in21k_ft_in1k" / "model.safetensors",),
            checkpoint_root / "maxvit_large",
        ),
        ModelSource(
            "dinov3_vit7b16",
            "DINOv3 ViT-7B/16",
            (
                *tuple((facebook_root / "dinov3-vit7b16-pretrain-lvd1689m").glob("*.safetensors")),
                facebook_root
                / "dinov3_vit7b16_imagenet1k_linear_head"
                / "dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth",
            ),
            checkpoint_root / "dinov3_vit7b16",
        ),
    ]


def _collect_rows(sources: list[ModelSource]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sources:
        dense_bytes = sum(_path_bytes(path) for path in source.dense_paths)
        if dense_bytes <= 0:
            continue
        for checkpoint in sorted(source.checkpoint_root.glob("*/model.pt")):
            checkpoint_bytes = checkpoint.stat().st_size
            ratio = dense_bytes / checkpoint_bytes if checkpoint_bytes > 0 else 0.0
            checkpoint_name = checkpoint.parent.name
            rows.append(
                {
                    "model": source.label,
                    "model_key": source.key,
                    "checkpoint": checkpoint_name,
                    "checkpoint_type": _checkpoint_type(checkpoint_name),
                    "dense_bytes": dense_bytes,
                    "checkpoint_bytes": checkpoint_bytes,
                    "file_size_ratio": ratio,
                    "dense_gib": dense_bytes / (1024**3),
                    "checkpoint_gib": checkpoint_bytes / (1024**3),
                    "path": str(checkpoint),
                    "note": _note(checkpoint_name),
                }
            )
    return rows


def _checkpoint_type(name: str) -> str:
    if name.startswith("cutlass_"):
        return "real_packed_checkpoint"
    return "fake_dense_state_dict"


def _note(name: str) -> str:
    if name.startswith("cutlass_"):
        return "Real packed/runtime/storage checkpoint; file-size ratio is meaningful."
    if "4over6" in name:
        return "Rescale fake-quant checkpoint stores dense tensors; not a real storage-compressed checkpoint."
    if name.startswith("nvfp4") or name.endswith("_sparse") or "sparse" in name:
        return "Fake/pruned checkpoint stores dense tensors; file-size ratio is close to 1x and not storage compression."
    return "Checkpoint file ratio computed from bytes."


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_key",
        "checkpoint",
        "checkpoint_type",
        "dense_bytes",
        "checkpoint_bytes",
        "dense_gib",
        "checkpoint_gib",
        "file_size_ratio",
        "path",
        "note",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["dense_gib"] = f"{row['dense_gib']:.3f}"
            out["checkpoint_gib"] = f"{row['checkpoint_gib']:.3f}"
            out["file_size_ratio"] = f"{row['file_size_ratio']:.3f}"
            writer.writerow(out)


def _write_md(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Checkpoint File Compression Ratios",
        "",
        "This report computes real file-size ratios as:",
        "",
        "`dense source model bytes / checkpoint model.pt bytes`",
        "",
        "Important: fake-quant/pruned checkpoints in this project usually store dense tensors in `state_dict`, so their real file-size ratio is close to `1x`. Only packed/runtime/storage checkpoints such as `cutlass_*` represent real storage-compressed checkpoint files.",
        "",
    ]
    for model in _ordered_models(rows):
        model_rows = [row for row in rows if row["model"] == model]
        lines.extend([f"## {model}", ""])
        lines.append("| Checkpoint | Type | Dense GiB | Checkpoint GiB | File Ratio | Note |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in sorted(model_rows, key=lambda r: (str(r["checkpoint_type"]), str(r["checkpoint"]))):
            lines.append(
                "| "
                f"`{row['checkpoint']}` | "
                f"{row['checkpoint_type']} | "
                f"{row['dense_gib']:.3f} | "
                f"{row['checkpoint_gib']:.3f} | "
                f"{row['file_size_ratio']:.3f}x | "
                f"{row['note']} |"
            )
        lines.append("")
    output.write_text("\n".join(lines) + "\n")


def _ordered_models(rows: list[dict[str, object]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        model = str(row["model"])
        if model not in seen:
            seen.append(model)
    return seen


def _path_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


if __name__ == "__main__":
    main()
