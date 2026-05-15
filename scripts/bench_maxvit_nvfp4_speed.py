#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch

from fake.evaluation.speed import benchmark_forward
from fake.kernels.flashinfer_nvfp4 import (
    FlashInferNVFP4Config,
    count_flashinfer_nvfp4_fallbacks,
    flashinfer_version,
)
from fake.models.maxvit import MAXVIT_VARIANT_CHOICES, get_maxvit_variant, maxvit_input_size, model_input_dtype
from fake.models.maxvit_nvfp4 import load_maxvit_flashinfer_nvfp4
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MaxViT with FlashInfer NVFP4 Linear kernels.")
    parser.add_argument("--variant", choices=MAXVIT_VARIANT_CHOICES, default="tiny")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-size", type=int, nargs=3, default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--output", default=None)
    parser.add_argument("--gemm-backend", choices=["auto", "cutlass", "cudnn", "trtllm", "cute-dsl", "b12x"], default="auto")
    parser.add_argument("--quant-backend", choices=["cuda", "cute-dsl"], default="cuda")
    parser.add_argument("--out-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--sf-layout", choices=["layout_128x4", "layout_8x4", "layout_linear"], default="layout_128x4")
    parser.add_argument("--per-token-activation", action="store_true")
    parser.add_argument("--fallback-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    variant_info = get_maxvit_variant(args.variant)
    output = args.output or f"artifacts/results/{variant_info.result_key}_nvfp4/speed.csv"
    nvfp4_config = FlashInferNVFP4Config(
        sf_layout=args.sf_layout,
        gemm_backend=args.gemm_backend,
        quant_backend=args.quant_backend,
        out_dtype=args.out_dtype,
        per_token_activation=args.per_token_activation,
        fallback_on_error=args.fallback_on_error,
    )
    model, config, report = load_maxvit_flashinfer_nvfp4(
        args.model_path,
        dtype=args.dtype,
        device=device,
        variant=args.variant,
        nvfp4_config=nvfp4_config,
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
    gpu_name = torch.cuda.get_device_name(device)
    kernel_fields = report.csv_fields()
    kernel_fields["fallback_count"] = count_flashinfer_nvfp4_fallbacks(model)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": variant_info.model_id,
        "model_variant": variant_info.variant,
        "method": "nvfp4_flashinfer",
        "task": "forward_speed",
        "speed_scope": "random_input_forward_only",
        "dtype_arg": args.dtype,
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": gpu_name,
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
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_module": os.environ.get("CUDA_MODULE", ""),
        "flashinfer_version": flashinfer_version(),
        **kernel_fields,
    }
    append_csv_row(output, list(row.keys()), row)
    print(
        "nvfp4 speed done: "
        f"variant={args.variant} batch_size={args.batch_size} dtype={row['runtime_dtype']} "
        f"replaced={report.replaced_linear_count} skipped={report.skipped_linear_count} "
        f"fallbacks={row['fallback_count']} mean_ms={row['latency_mean_ms']} "
        f"images_per_sec={row['images_per_sec']} output={output}"
    )
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


if __name__ == "__main__":
    main()
