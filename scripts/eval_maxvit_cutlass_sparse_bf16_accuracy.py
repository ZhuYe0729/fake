#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from fake.compression.checkpoint import checkpoint_csv_fields
from fake.data.imagenet_zip import DEFAULT_IMAGENET_ROOT, ImageNetZipDataset
from fake.evaluation.accuracy import evaluate_topk
from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config, count_cutlass_sparse_bf16_modules
from fake.models.dinov3_cutlass_runtime import runtime_checkpoint_csv_fields
from fake.models.maxvit import MAXVIT_VARIANT_CHOICES, get_maxvit_variant, model_input_dtype
from fake.models.maxvit_cutlass_checkpoint import load_maxvit_cutlass_sparse_bf16_runtime
from fake.models.maxvit_cutlass_sparse_bf16 import load_maxvit_cutlass_sparse_bf16
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MaxViT CUTLASS sparse BF16 on ImageNet val zip dataset.")
    parser.add_argument("--variant", choices=MAXVIT_VARIANT_CHOICES, default="tiny")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-root", default=str(DEFAULT_IMAGENET_ROOT))
    parser.add_argument("--csv", default="val.csv")
    parser.add_argument("--zip", default="imagenet_val.zip")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--runtime-checkpoint", default=None)
    parser.add_argument("--no-prune", action="store_true", help="Require weights to already satisfy 2:4 sparsity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    variant_info = get_maxvit_variant(args.variant)
    output = args.output or f"artifacts/results/{variant_info.result_key}_cutlass_sparse_bf16/accuracy.csv"
    runtime_load = None
    if args.runtime_checkpoint:
        model, config, report, runtime_load = load_maxvit_cutlass_sparse_bf16_runtime(
            args.runtime_checkpoint,
            model_path=args.model_path,
            device=device,
            variant=args.variant,
        )
        checkpoint_metadata = runtime_load.metadata.get("source_checkpoint_metadata", {})
    else:
        model, config, report, checkpoint_metadata = load_maxvit_cutlass_sparse_bf16(
            model_path=args.model_path,
            device=device,
            variant=args.variant,
            sparse_config=CutlassSparseBF16Config(prune=not args.no_prune),
            checkpoint_path=args.checkpoint,
        )
    input_dtype = model_input_dtype(model)
    dataset = ImageNetZipDataset(args.dataset_root, args.csv, args.zip, config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    result = evaluate_topk(model, dataloader, device, input_dtype, args.log_interval)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": variant_info.model_id,
        "model_variant": variant_info.variant,
        "method": "sparse_bf16_cutlass",
        "task": "imagenet_accuracy",
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device),
        "num_samples": result.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "top1": f"{result.top1:.6f}",
        "top5": f"{result.top5:.6f}",
        "elapsed_sec": f"{result.elapsed_sec:.3f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "model_path": args.model_path or str(variant_info.model_path),
        "dataset_root": args.dataset_root,
        "csv": args.csv,
        "zip": args.zip,
        "cutlass_sparse_bf16_module_count": count_cutlass_sparse_bf16_modules(model),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **checkpoint_csv_fields(
            checkpoint_metadata,
            args.checkpoint or (runtime_load.metadata.get("source_checkpoint_path", "") if runtime_load else None),
            "sparse_bf16_cutlass",
        ),
        **runtime_checkpoint_csv_fields(runtime_load),
        **report.csv_fields(),
    }
    append_csv_row(output, list(row.keys()), row)
    print(
        "maxvit cutlass sparse bf16 accuracy done: "
        f"variant={args.variant} top1={row['top1']} top5={row['top5']} "
        f"replaced={report.replaced_linear_count} skipped={report.skipped_linear_count} output={output}"
    )
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


if __name__ == "__main__":
    main()
