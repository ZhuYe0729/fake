#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch

from fake.evaluation.speed import benchmark_forward
from fake.models.maxvit import DEFAULT_MAXVIT_MODEL_PATH, load_maxvit_dense, model_input_dtype
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense MaxViT forward speed.")
    parser.add_argument("--model-path", default=str(DEFAULT_MAXVIT_MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-size", type=int, nargs=3, default=[3, 224, 224])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--output", default="artifacts/results/maxvit_dense/speed.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    model, _ = load_maxvit_dense(args.model_path, dtype=args.dtype, device=device)
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
    gpu_name = torch.cuda.get_device_name(device)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "timm/maxvit_tiny_tf_224.in1k",
        "method": "dense",
        "task": "forward_speed",
        "speed_scope": "random_input_forward_only",
        "dtype_arg": args.dtype,
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": gpu_name,
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
        "model_path": args.model_path,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    fieldnames = list(row.keys())
    append_csv_row(args.output, fieldnames, row)
    print(
        "speed done: "
        f"batch_size={args.batch_size} dtype={row['runtime_dtype']} "
        f"mean_ms={row['latency_mean_ms']} images_per_sec={row['images_per_sec']} "
        f"warmup={args.warmup} iters={args.iters} output={args.output}"
    )


if __name__ == "__main__":
    main()

