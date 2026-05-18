#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fake.compression.activation import apply_dinov3_activation_fake_quant
from fake.compression.checkpoint import checkpoint_csv_fields, load_checkpoint_into_model
from fake.data.dinov3_transforms import build_dinov3_lvd1689m_transform
from fake.data.imagenet_zip import DEFAULT_IMAGENET_ROOT, ImageNetZipDataset
from fake.evaluation.accuracy import evaluate_topk
from fake.models.dinov3 import (
    DEFAULT_DINOV3_BACKBONE_PATH,
    DEFAULT_DINOV3_HEAD_PATH,
    load_dinov3_vit7b16_dense_classifier,
    model_input_dtype,
)
from fake.utils.csv_io import append_csv_row


RESULT_DIRS = {
    "nvfp4_4over6_unstructured_sparse": "artifacts/results/dinov3_vit7b16_4over6_unstructured_sparse",
    "nvfp4_4over6_semi_structured_sparse": "artifacts/results/dinov3_vit7b16_4over6_semi_structured_sparse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 ViT-7B/16 Four Over Six fake-quant checkpoints.")
    parser.add_argument("--method", choices=sorted(RESULT_DIRS), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_IMAGENET_ROOT))
    parser.add_argument("--csv", default="val.csv")
    parser.add_argument("--zip", default="imagenet_val.zip")
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-activation-quant", action="store_true")
    parser.add_argument("--activation-group-size", type=int, default=None)
    parser.add_argument("--activation-scale-rule", choices=["static_6", "four_over_six_mse"], default="four_over_six_mse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    checkpoint = args.checkpoint or f"artifacts/checkpoints/dinov3_vit7b16/{args.method}/model.pt"
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output = args.output or f"{RESULT_DIRS[args.method]}/accuracy.csv"

    device = torch.device("cuda")
    model, config = load_dinov3_vit7b16_dense_classifier(args.backbone_path, args.head_path, device=device)
    checkpoint_metadata = load_checkpoint_into_model(model, checkpoint)
    activation_group_size = args.activation_group_size or int(checkpoint_metadata.get("nvfp4_group_size", 16) or 16)
    activation_modules = 0
    if not args.no_activation_quant:
        activation_modules = apply_dinov3_activation_fake_quant(
            model,
            group_size=activation_group_size,
            scale_rule=args.activation_scale_rule,
        )

    input_dtype = model_input_dtype(model)
    dataset = ImageNetZipDataset(
        args.dataset_root,
        args.csv,
        args.zip,
        transform=build_dinov3_lvd1689m_transform(args.resize_size),
    )
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
        "model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "head": "dinov3_vit7b16_imagenet1k_linear_head",
        "method": args.method,
        "task": "imagenet_accuracy",
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device),
        "num_samples": result.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "resize_size": args.resize_size,
        "hidden_size": config.get("hidden_size", ""),
        "num_register_tokens": config.get("num_register_tokens", ""),
        "top1": f"{result.top1:.6f}",
        "top5": f"{result.top5:.6f}",
        "elapsed_sec": f"{result.elapsed_sec:.3f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "backbone_path": args.backbone_path,
        "head_path": args.head_path,
        "dataset_root": args.dataset_root,
        "csv": args.csv,
        "zip": args.zip,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **checkpoint_csv_fields(checkpoint_metadata, checkpoint, args.method),
        "activation_quant": not args.no_activation_quant,
        "activation_quant_modules": activation_modules,
        "activation_group_size": activation_group_size if not args.no_activation_quant else "",
        "activation_scale_rule": args.activation_scale_rule if not args.no_activation_quant else "",
    }
    append_csv_row(output, list(row.keys()), row)
    print(
        "dinov3 four_over_six accuracy done: "
        f"method={args.method} top1={row['top1']} top5={row['top5']} samples={result.num_samples} "
        f"activation_quant={row['activation_quant']} activation_modules={activation_modules} output={output}"
    )


if __name__ == "__main__":
    main()
