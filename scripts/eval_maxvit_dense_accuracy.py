#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from fake.compression.checkpoint import checkpoint_csv_fields, load_checkpoint_into_model
from fake.data.imagenet_zip import DEFAULT_IMAGENET_ROOT, ImageNetZipDataset
from fake.evaluation.accuracy import evaluate_topk
from fake.models.maxvit import DEFAULT_MAXVIT_MODEL_PATH, load_maxvit_dense, model_input_dtype
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense MaxViT on ImageNet val zip dataset.")
    parser.add_argument("--model-path", default=str(DEFAULT_MAXVIT_MODEL_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_IMAGENET_ROOT))
    parser.add_argument("--csv", default="val.csv")
    parser.add_argument("--zip", default="imagenet_val.zip")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output", default="artifacts/results/maxvit_dense/accuracy.csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--method", default="dense")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    model, config = load_maxvit_dense(args.model_path, dtype=args.dtype, device=device)
    checkpoint_metadata = load_checkpoint_into_model(model, args.checkpoint)
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
    gpu_name = torch.cuda.get_device_name(device)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "timm/maxvit_tiny_tf_224.in1k",
        "method": args.method,
        "task": "imagenet_accuracy",
        "dtype_arg": args.dtype,
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": gpu_name,
        "num_samples": result.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "top1": f"{result.top1:.6f}",
        "top5": f"{result.top5:.6f}",
        "elapsed_sec": f"{result.elapsed_sec:.3f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "model_path": args.model_path,
        "dataset_root": args.dataset_root,
        "csv": args.csv,
        "zip": args.zip,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **checkpoint_csv_fields(checkpoint_metadata, args.checkpoint, args.method),
    }
    fieldnames = list(row.keys())
    append_csv_row(args.output, fieldnames, row)
    print(
        "accuracy done: "
        f"top1={row['top1']} top5={row['top5']} samples={result.num_samples} "
        f"batch_size={args.batch_size} dtype={row['runtime_dtype']} output={args.output}"
    )


if __name__ == "__main__":
    main()
