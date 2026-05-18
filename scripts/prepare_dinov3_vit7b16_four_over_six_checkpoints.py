#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys

from fake.compression.pipeline import default_nvfp4_group_size


FOUR_OVER_SIX_METHODS = (
    "nvfp4_4over6_unstructured_sparse",
    "nvfp4_4over6_semi_structured_sparse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DINOv3 ViT-7B/16 Four Over Six fake-quant checkpoints.")
    parser.add_argument("--methods", nargs="+", choices=FOUR_OVER_SIX_METHODS, default=list(FOUR_OVER_SIX_METHODS))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--calib-samples", type=int, default=None)
    parser.add_argument("--calib-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--save-full-masks", action="store_true")
    parser.add_argument("--save-full-scales", action="store_true")
    parser.add_argument("--dinov3-backbone-path", default=None)
    parser.add_argument("--dinov3-head-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for method in args.methods:
        cmd = [
            sys.executable,
            "scripts/prepare_compressed_model.py",
            "--model",
            "dinov3_vit7b16",
            "--method",
            method,
            "--nvfp4-scale-rule",
            "four_over_six_mse",
            "--nvfp4-group-size",
            str(default_nvfp4_group_size(method)),
        ]
        _append_optional(cmd, "--dataset-root", args.dataset_root)
        _append_optional(cmd, "--csv", args.csv)
        _append_optional(cmd, "--zip", args.zip)
        _append_optional(cmd, "--calib-samples", args.calib_samples)
        _append_optional(cmd, "--calib-batch-size", args.calib_batch_size)
        _append_optional(cmd, "--num-workers", args.num_workers)
        _append_optional(cmd, "--sparsity", args.sparsity)
        _append_optional(cmd, "--dinov3-backbone-path", args.dinov3_backbone_path)
        _append_optional(cmd, "--dinov3-head-path", args.dinov3_head_path)
        if args.save_full_masks:
            cmd.append("--save-full-masks")
        if args.save_full_scales:
            cmd.append("--save-full-scales")
        subprocess.run(cmd, check=True)


def _append_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


if __name__ == "__main__":
    main()
