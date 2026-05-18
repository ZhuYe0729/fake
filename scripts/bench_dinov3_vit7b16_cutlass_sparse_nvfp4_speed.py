#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch

from fake.compression.checkpoint import checkpoint_csv_fields
from fake.evaluation.speed import benchmark_forward
from fake.kernels.cutlass_sparse_nvfp4 import (
    CutlassSparseNVFP4Config,
    count_cutlass_sparse_nvfp4_modules,
)
from fake.models.dinov3 import DEFAULT_DINOV3_BACKBONE_PATH, DEFAULT_DINOV3_HEAD_PATH, model_input_dtype
from fake.models.dinov3_cutlass_runtime import (
    load_dinov3_vit7b16_cutlass_runtime_classifier,
    runtime_checkpoint_csv_fields,
)
from fake.models.dinov3_cutlass_sparse_nvfp4 import load_dinov3_vit7b16_cutlass_sparse_nvfp4_classifier
from fake.models.dinov3_cutlass_storage import load_dinov3_vit7b16_cutlass_storage_classifier
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DINOv3 ViT-7B/16 CUTLASS sparse NVFP4 classifier forward speed.")
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, nargs=3, default=[3, 256, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default="artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/speed.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--runtime-checkpoint", default=None)
    parser.add_argument("--storage-checkpoint", default=None)
    parser.add_argument("--no-prune", action="store_true", help="Require weights to already satisfy sparse pattern.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    runtime_load = None
    checkpoint_metadata = {}
    if args.runtime_checkpoint and args.storage_checkpoint:
        raise ValueError("--runtime-checkpoint and --storage-checkpoint are mutually exclusive")
    if args.storage_checkpoint:
        model, config, report, runtime_load = load_dinov3_vit7b16_cutlass_storage_classifier(
            storage_checkpoint_path=args.storage_checkpoint,
            backbone_path=args.backbone_path,
            head_path=args.head_path,
            device=device,
        )
        checkpoint_metadata = runtime_load.metadata.get("source_checkpoint_metadata", {})
    elif args.runtime_checkpoint:
        model, config, report, runtime_load = load_dinov3_vit7b16_cutlass_runtime_classifier(
            runtime_checkpoint_path=args.runtime_checkpoint,
            backbone_path=args.backbone_path,
            head_path=args.head_path,
            device=device,
        )
        if runtime_load.metadata.get("backend") != "sparse_nvfp4":
            raise ValueError(f"Expected sparse_nvfp4 runtime checkpoint, got {runtime_load.metadata.get('backend')}")
        checkpoint_metadata = runtime_load.metadata.get("source_checkpoint_metadata", {})
    else:
        nvfp4_config = CutlassSparseNVFP4Config(prune=not args.no_prune)
        model, config, report, checkpoint_metadata = load_dinov3_vit7b16_cutlass_sparse_nvfp4_classifier(
            backbone_path=args.backbone_path,
            head_path=args.head_path,
            device=device,
            nvfp4_config=nvfp4_config,
            checkpoint_path=args.checkpoint,
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
        "method": "sparse_nvfp4_cutlass",
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
        "cutlass_sparse_nvfp4_module_count": count_cutlass_sparse_nvfp4_modules(model),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **checkpoint_csv_fields(
            checkpoint_metadata,
            args.checkpoint or (runtime_load.metadata.get("source_checkpoint_path", "") if runtime_load else None),
            "sparse_nvfp4_cutlass",
        ),
        **runtime_checkpoint_csv_fields(runtime_load),
        **report.csv_fields(),
    }
    append_csv_row(args.output, list(row.keys()), row)
    print(
        "dinov3 cutlass sparse nvfp4 speed done: "
        f"batch_size={args.batch_size} dtype={row['runtime_dtype']} "
        f"replaced={report.replaced_linear_count} skipped={report.skipped_linear_count} "
        f"mean_ms={row['latency_mean_ms']} images_per_sec={row['images_per_sec']} "
        f"warmup={args.warmup} iters={args.iters} output={args.output}"
    )
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


if __name__ == "__main__":
    main()
