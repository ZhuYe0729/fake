#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.compression.checkpoint import checkpoint_csv_fields
from fake.evaluation.speed import benchmark_forward
from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config, replace_linear_with_cutlass_nvfp4
from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config, replace_linear_with_cutlass_sparse_bf16
from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config, replace_linear_with_cutlass_sparse_nvfp4
from fake.kernels.marlin_nvfp4 import load_marlin_nvfp4_checkpoint_into_model
from fake.models.mirror import (
    DEFAULT_MIRROR_BACKBONE_PATH,
    DEFAULT_MIRROR_MEMORY_PATH,
    DEFAULT_MIRROR_MODEL_PATH,
    load_mirror_compressed_detector,
    load_mirror_dense_detector,
    model_input_dtype,
)
from fake.utils.csv_io import append_csv_row


RUNTIME_METHODS = ("dense", "nvfp4", "marlin_nvfp4", "semi_structured_sparse", "nvfp4_semi_structured_sparse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MIRROR compressed detector forward speed.")
    parser.add_argument("--method", choices=RUNTIME_METHODS, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_MIRROR_MODEL_PATH))
    parser.add_argument("--memory-path", default=str(DEFAULT_MIRROR_MEMORY_PATH))
    parser.add_argument("--backbone-path", default=str(DEFAULT_MIRROR_BACKBONE_PATH))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, nargs=3, default=[3, 224, 224])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default="artifacts/results/mirror_compressed/speed.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    checkpoint_metadata: dict[str, object] = {}
    if args.method == "dense":
        model, config = load_mirror_dense_detector(
            model_path=args.model_path,
            memory_path=args.memory_path,
            backbone_path=args.backbone_path,
            device=device,
        )
        report_fields = {"kernel_backend": "torch_dense", "replaced_linear_count": "", "skipped_linear_count": ""}
    elif args.method == "marlin_nvfp4":
        checkpoint = args.checkpoint or "artifacts/checkpoints/mirror/marlin_nvfp4/model.pt"
        model, config = load_mirror_dense_detector(
            model_path=args.model_path,
            memory_path=args.memory_path,
            backbone_path=args.backbone_path,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        checkpoint_metadata, report = load_marlin_nvfp4_checkpoint_into_model(model, checkpoint, device=device)
        args.checkpoint = checkpoint
        report_fields = report.csv_fields()
        if report.skipped:
            print(f"skipped_modules={report.skipped[:10]}")
    else:
        checkpoint = args.checkpoint or f"artifacts/checkpoints/mirror/{args.method}/model.pt"
        model, config, checkpoint_metadata = load_mirror_compressed_detector(
            checkpoint,
            model_path=args.model_path,
            memory_path=args.memory_path,
            backbone_path=args.backbone_path,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        args.checkpoint = checkpoint
        if args.method == "nvfp4":
            report = replace_linear_with_cutlass_nvfp4(model, "mirror", CutlassNVFP4Config())
        elif args.method == "semi_structured_sparse":
            report = replace_linear_with_cutlass_sparse_bf16(
                model,
                "mirror",
                CutlassSparseBF16Config(prune=False),
            )
        elif args.method == "nvfp4_semi_structured_sparse":
            report = replace_linear_with_cutlass_sparse_nvfp4(
                model,
                "mirror",
                CutlassSparseNVFP4Config(prune=False),
            )
        else:
            raise ValueError(f"Unsupported MIRROR runtime method: {args.method}")
        report_fields = report.csv_fields()
        if report.skipped:
            print(f"skipped_modules={report.skipped[:10]}")

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
        "model": "MIRROR-DINOv3-Huge",
        "method": args.method,
        "task": "forward_speed",
        "speed_scope": "random_input_detector_forward_only",
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "input_c": args.input_size[0],
        "input_h": args.input_size[1],
        "input_w": args.input_size[2],
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_mean_ms": f"{result.latency_mean_ms:.6f}",
        "latency_p50_ms": f"{result.latency_p50_ms:.6f}",
        "latency_p90_ms": f"{result.latency_p90_ms:.6f}",
        "latency_min_ms": f"{result.latency_min_ms:.6f}",
        "latency_max_ms": f"{result.latency_max_ms:.6f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "model_path": config.get("model_path", args.model_path),
        "memory_path": config.get("memory_path", args.memory_path),
        "backbone_path": config.get("backbone_path", args.backbone_path),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **checkpoint_csv_fields(checkpoint_metadata, args.checkpoint, args.method),
        **report_fields,
    }
    append_csv_row(args.output, list(row.keys()), row)
    print(
        "mirror speed done: "
        f"method={args.method} batch_size={args.batch_size} dtype={row['runtime_dtype']} "
        f"mean_ms={row['latency_mean_ms']} images_per_sec={row['images_per_sec']} output={args.output}"
    )


if __name__ == "__main__":
    main()
