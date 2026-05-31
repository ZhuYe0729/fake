#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch

from fake.evaluation.speed import benchmark_forward
from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config
from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config
from fake.models.dinov3 import DEFAULT_DINOV3_BACKBONE_PATH, DEFAULT_DINOV3_HEAD_PATH, model_input_dtype
from fake.models.dinov3_cutlass_hybrid import (
    HybridScheme,
    load_dinov3_vit7b16_cutlass_hybrid_classifier,
)
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DINOv3 ViT-7B/16 CUTLASS hybrid classifier forward speed.")
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, nargs=3, default=[3, 256, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default="artifacts/results/dinov3_vit7b16_cutlass_hybrid/speed.csv")
    parser.add_argument("--hybrid-scheme", choices=["b16_manual", "b32_manual"], default="b16_manual")
    parser.add_argument("--no-prune", action="store_true", help="Require weights to already satisfy sparse patterns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    model, config, report = load_dinov3_vit7b16_cutlass_hybrid_classifier(
        backbone_path=args.backbone_path,
        head_path=args.head_path,
        device=device,
        hybrid_scheme=args.hybrid_scheme,
        sparse_nvfp4_config=CutlassSparseNVFP4Config(prune=not args.no_prune),
        sparse_bf16_config=CutlassSparseBF16Config(prune=not args.no_prune),
    )
    input_dtype = model_input_dtype(model)
    result = benchmark_forward(
        model=model,
        batch_size=args.batch_size,
        input_size=tuple(args.input_size),
        input_dtype=input_dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "head": "dinov3_vit7b16_imagenet1k_linear_head",
        "method": "hybrid_cutlass",
        "hybrid_scheme": args.hybrid_scheme,
        "task": "forward_speed",
        "speed_scope": "random_input_classifier_forward_only",
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "input_c": args.input_size[0],
        "input_h": args.input_size[1],
        "input_w": args.input_size[2],
        "warmup": args.warmup,
        "iters": args.iters,
        "hidden_size": config.get("hidden_size", ""),
        "num_register_tokens": config.get("num_register_tokens", ""),
        "latency_mean_ms": f"{result.latency_mean_ms:.6f}",
        "latency_p50_ms": f"{result.latency_p50_ms:.6f}",
        "latency_p90_ms": f"{result.latency_p90_ms:.6f}",
        "latency_min_ms": f"{result.latency_min_ms:.6f}",
        "latency_max_ms": f"{result.latency_max_ms:.6f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "backbone_path": args.backbone_path,
        "head_path": args.head_path,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **report.csv_fields(),
    }
    append_csv_row(args.output, list(row.keys()), row)
    print(
        "dinov3 cutlass hybrid speed done: "
        f"scheme={args.hybrid_scheme} batch_size={args.batch_size} dtype={row['runtime_dtype']} "
        f"replaced={report.replaced_linear_count} skipped={report.skipped_linear_count} "
        f"sparse_nvfp4={report.sparse_nvfp4_module_count} sparse_bf16={report.sparse_bf16_module_count} "
        f"mean_ms={row['latency_mean_ms']} images_per_sec={row['images_per_sec']} "
        f"warmup={args.warmup} iters={args.iters} output={args.output}"
    )
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


if __name__ == "__main__":
    main()
