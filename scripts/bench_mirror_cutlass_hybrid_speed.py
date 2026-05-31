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

from fake.evaluation.speed import benchmark_forward
from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config
from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config
from fake.models.mirror import (
    DEFAULT_MIRROR_BACKBONE_PATH,
    DEFAULT_MIRROR_MEMORY_PATH,
    DEFAULT_MIRROR_MODEL_PATH,
    model_input_dtype,
)
from fake.models.mirror_cutlass_hybrid import HYBRID_SCHEMES, load_mirror_cutlass_hybrid_detector
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MIRROR CUTLASS hybrid detector forward speed.")
    parser.add_argument("--model-path", default=str(DEFAULT_MIRROR_MODEL_PATH))
    parser.add_argument("--memory-path", default=str(DEFAULT_MIRROR_MEMORY_PATH))
    parser.add_argument("--backbone-path", default=str(DEFAULT_MIRROR_BACKBONE_PATH))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, nargs=3, default=[3, 224, 224])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default="artifacts/results/mirror_cutlass_hybrid/speed.csv")
    parser.add_argument("--hybrid-scheme", choices=HYBRID_SCHEMES, default="dino_b32_like")
    parser.add_argument("--no-prune", action="store_true", help="Require weights to already satisfy sparse patterns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    model, config, report = load_mirror_cutlass_hybrid_detector(
        model_path=args.model_path,
        memory_path=args.memory_path,
        backbone_path=args.backbone_path,
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
        "model": "MIRROR-DINOv3-Huge",
        "method": "hybrid_cutlass",
        "hybrid_scheme": args.hybrid_scheme,
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
        **report.csv_fields(),
    }
    append_csv_row(args.output, list(row.keys()), row)
    print(
        "mirror cutlass hybrid speed done: "
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
