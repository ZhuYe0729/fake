#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch

from fake.evaluation.speed import benchmark_forward
from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config, count_cutlass_nvfp4_modules
from fake.models.maxvit import MAXVIT_VARIANT_CHOICES, get_maxvit_variant, maxvit_input_size, model_input_dtype
from fake.models.dinov3_cutlass_runtime import runtime_checkpoint_csv_fields
from fake.models.maxvit_cutlass_checkpoint import load_maxvit_cutlass_dense_runtime
from fake.models.maxvit_cutlass_nvfp4 import load_maxvit_cutlass_nvfp4
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MaxViT with CUTLASS dense NVFP4 Linear kernels.")
    parser.add_argument("--variant", choices=MAXVIT_VARIANT_CHOICES, default="tiny")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-size", type=int, nargs=3, default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", default=None)
    parser.add_argument("--runtime-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    variant_info = get_maxvit_variant(args.variant)
    output = args.output or f"artifacts/results/{variant_info.result_key}_cutlass_nvfp4/speed.csv"
    runtime_load = None
    if args.runtime_checkpoint:
        model, config, report, runtime_load = load_maxvit_cutlass_dense_runtime(
            args.runtime_checkpoint,
            model_path=args.model_path,
            device=device,
            variant=args.variant,
        )
    else:
        model, config, report = load_maxvit_cutlass_nvfp4(
            model_path=args.model_path,
            device=device,
            variant=args.variant,
            nvfp4_config=CutlassNVFP4Config(),
        )
    input_dtype = model_input_dtype(model)
    input_size = tuple(args.input_size) if args.input_size is not None else maxvit_input_size(config)
    result = benchmark_forward(
        model=model,
        batch_size=args.batch_size,
        input_size=input_size,
        input_dtype=input_dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": variant_info.model_id,
        "model_variant": variant_info.variant,
        "method": "nvfp4_cutlass",
        "task": "forward_speed",
        "speed_scope": "random_input_forward_only",
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "input_c": input_size[0],
        "input_h": input_size[1],
        "input_w": input_size[2],
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_mean_ms": f"{result.latency_mean_ms:.6f}",
        "latency_p50_ms": f"{result.latency_p50_ms:.6f}",
        "latency_p90_ms": f"{result.latency_p90_ms:.6f}",
        "latency_min_ms": f"{result.latency_min_ms:.6f}",
        "latency_max_ms": f"{result.latency_max_ms:.6f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "model_path": args.model_path or str(variant_info.model_path),
        "cutlass_nvfp4_module_count": count_cutlass_nvfp4_modules(model),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **runtime_checkpoint_csv_fields(runtime_load),
        **report.csv_fields(),
    }
    append_csv_row(output, list(row.keys()), row)
    print(
        "maxvit cutlass nvfp4 speed done: "
        f"variant={args.variant} batch_size={args.batch_size} dtype={row['runtime_dtype']} "
        f"replaced={report.replaced_linear_count} skipped={report.skipped_linear_count} "
        f"mean_ms={row['latency_mean_ms']} images_per_sec={row['images_per_sec']} output={output}"
    )
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


if __name__ == "__main__":
    main()
