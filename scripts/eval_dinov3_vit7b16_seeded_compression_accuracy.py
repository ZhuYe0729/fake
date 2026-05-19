#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from fake.compression.activation import apply_dinov3_activation_fake_quant
from fake.compression.checkpoint import checkpoint_csv_fields
from fake.compression.pipeline import (
    SUPPORTED_METHODS,
    CompressionConfig,
    compress_model,
    default_calib_batch_size,
    default_calib_samples,
    default_nvfp4_group_size,
)
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


METHODS = (
    "nvfp4_unstructured_sparse",
    "nvfp4_semi_structured_sparse",
    "nvfp4_4over6_unstructured_sparse",
    "nvfp4_4over6_semi_structured_sparse",
)
FOUR_OVER_SIX_METHODS = {
    "nvfp4_4over6_unstructured_sparse",
    "nvfp4_4over6_semi_structured_sparse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate seeded DINOv3 compressed candidates without saving checkpoints."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calib-shuffle", action="store_true")
    parser.add_argument("--calib-samples", type=int, default=None)
    parser.add_argument("--calib-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--nvfp4-group-size", type=int, default=None)
    parser.add_argument("--nvfp4-scale-precision", default="fp16")
    parser.add_argument("--nvfp4-scale-rule", choices=["static_6", "four_over_six_mse"], default=None)
    parser.add_argument("--nvfp4-scale-remap", default="none")
    parser.add_argument("--activation-mode", choices=["auto", "off", "on", "both"], default="auto")
    parser.add_argument("--activation-group-size", type=int, default=None)
    parser.add_argument("--activation-scale-rule", choices=["static_6", "four_over_six_mse"], default="four_over_six_mse")
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_IMAGENET_ROOT))
    parser.add_argument("--csv", default="val.csv")
    parser.add_argument("--zip", default="imagenet_val.zip")
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported compression method: {args.method}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    model, model_config = load_dinov3_vit7b16_dense_classifier(
        args.backbone_path,
        args.head_path,
        device=device,
    )
    input_dtype = model_input_dtype(model)
    calib_samples = args.calib_samples if args.calib_samples is not None else default_calib_samples("dinov3_vit7b16")
    calib_batch_size = (
        args.calib_batch_size
        if args.calib_batch_size is not None
        else default_calib_batch_size("dinov3_vit7b16")
    )
    group_size = args.nvfp4_group_size if args.nvfp4_group_size is not None else default_nvfp4_group_size(args.method)
    scale_rule = args.nvfp4_scale_rule or _default_nvfp4_scale_rule(args.method)

    dataset = ImageNetZipDataset(
        args.dataset_root,
        args.csv,
        args.zip,
        transform=build_dinov3_lvd1689m_transform(args.resize_size),
    )
    calib_loader = _build_loader(
        dataset=dataset,
        batch_size=calib_batch_size,
        num_workers=args.num_workers,
        shuffle=args.calib_shuffle,
        seed=args.seed,
    )
    eval_loader = _build_loader(
        dataset=dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
    )

    compression_config = CompressionConfig(
        model_name="dinov3_vit7b16",
        method=args.method,
        calib_samples=calib_samples,
        sparsity=args.sparsity,
        nvfp4_group_size=group_size,
        nvfp4_scale_precision=args.nvfp4_scale_precision,
        nvfp4_scale_rule=scale_rule,
        nvfp4_scale_remap=args.nvfp4_scale_remap,
    )
    metadata, _masks, _scales = compress_model(
        model=model,
        dataloader=calib_loader,
        device=device,
        input_dtype=input_dtype,
        config=compression_config,
    )
    metadata.update(
        {
            "checkpoint_path": "",
            "dataset_root": args.dataset_root,
            "csv": args.csv,
            "zip": args.zip,
            "calib_shuffle": args.calib_shuffle,
            "seed": args.seed,
        }
    )

    activation_wrapped = False
    activation_group_size = args.activation_group_size or group_size
    for activation_quant in _activation_modes(args.method, args.activation_mode):
        activation_modules = 0
        if activation_quant:
            if not activation_wrapped:
                activation_modules = apply_dinov3_activation_fake_quant(
                    model,
                    group_size=activation_group_size,
                    scale_rule=args.activation_scale_rule,
                )
                activation_wrapped = True
        else:
            row_activation_group_size = ""

        result = evaluate_topk(model, eval_loader, device, input_dtype, args.log_interval)
        output = args.output or _default_output(args.method)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            "head": "dinov3_vit7b16_imagenet1k_linear_head",
            "method": args.method,
            "task": "imagenet_accuracy",
            "runtime_dtype": str(input_dtype).replace("torch.", ""),
            "device": torch.cuda.get_device_name(device),
            "num_samples": result.num_samples,
            "batch_size": args.eval_batch_size,
            "num_workers": args.num_workers,
            "resize_size": args.resize_size,
            "hidden_size": model_config.get("hidden_size", ""),
            "num_register_tokens": model_config.get("num_register_tokens", ""),
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
            **checkpoint_csv_fields(metadata, None, args.method),
            "seed": args.seed,
            "calib_shuffle": args.calib_shuffle,
            "no_checkpoint_eval": True,
            "activation_quant": activation_quant,
            "activation_quant_modules": activation_modules,
            "activation_group_size": activation_group_size if activation_quant else row_activation_group_size,
            "activation_scale_rule": args.activation_scale_rule if activation_quant else "",
        }
        append_csv_row(output, list(row.keys()), row)
        print(
            "dinov3 seeded compression accuracy done: "
            f"method={args.method} seed={args.seed} top1={row['top1']} top5={row['top5']} "
            f"activation_quant={activation_quant} output={output}"
        )


def _build_loader(
    dataset: ImageNetZipDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _activation_modes(method: str, activation_mode: str) -> list[bool]:
    if activation_mode == "off":
        return [False]
    if activation_mode == "on":
        return [True]
    if activation_mode == "both":
        return [False, True]
    if method in FOUR_OVER_SIX_METHODS:
        return [False, True]
    return [False]


def _default_nvfp4_scale_rule(method: str) -> str:
    if method in FOUR_OVER_SIX_METHODS:
        return "four_over_six_mse"
    return "static_6"


def _default_output(method: str) -> str:
    if method == "nvfp4_4over6_unstructured_sparse":
        return "artifacts/results/dinov3_vit7b16_4over6_unstructured_sparse/accuracy_seeded.csv"
    if method == "nvfp4_4over6_semi_structured_sparse":
        return "artifacts/results/dinov3_vit7b16_4over6_semi_structured_sparse/accuracy_seeded.csv"
    return "artifacts/results/dinov3_vit7b16_compressed/accuracy_seeded.csv"


if __name__ == "__main__":
    main()
