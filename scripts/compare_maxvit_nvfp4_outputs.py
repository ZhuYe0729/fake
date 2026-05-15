#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from fake.kernels.flashinfer_nvfp4 import FlashInferNVFP4Config
from fake.models.maxvit import (
    MAXVIT_VARIANT_CHOICES,
    load_maxvit_dense,
    maxvit_input_size,
    model_input_dtype,
)
from fake.models.maxvit_nvfp4 import load_maxvit_flashinfer_nvfp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense MaxViT logits with FlashInfer NVFP4 MaxViT logits.")
    parser.add_argument("--variant", choices=MAXVIT_VARIANT_CHOICES, default="tiny")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, nargs=3, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--gemm-backend", choices=["auto", "cutlass", "cudnn", "trtllm", "cute-dsl", "b12x"], default="auto")
    parser.add_argument("--quant-backend", choices=["cuda", "cute-dsl"], default="cuda")
    parser.add_argument("--fallback-on-error", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    dense, config = load_maxvit_dense(args.model_path, dtype=args.dtype, device=device, variant=args.variant)
    nvfp4, _, report = load_maxvit_flashinfer_nvfp4(
        args.model_path,
        dtype=args.dtype,
        device=device,
        variant=args.variant,
        nvfp4_config=FlashInferNVFP4Config(
            gemm_backend=args.gemm_backend,
            quant_backend=args.quant_backend,
            fallback_on_error=args.fallback_on_error,
        ),
    )
    input_dtype = model_input_dtype(dense)
    input_size = tuple(args.input_size) if args.input_size is not None else maxvit_input_size(config)
    inputs = torch.randn((args.batch_size, *input_size), device=device, dtype=input_dtype)
    dense_logits = dense(inputs)
    nvfp4_logits = nvfp4(inputs)
    diff = (nvfp4_logits.float() - dense_logits.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        nvfp4_logits.float().reshape(args.batch_size, -1),
        dense_logits.float().reshape(args.batch_size, -1),
        dim=-1,
    )
    top1_agreement = (nvfp4_logits.argmax(dim=-1) == dense_logits.argmax(dim=-1)).float().mean().item()
    rmse = torch.sqrt(torch.mean((nvfp4_logits.float() - dense_logits.float()) ** 2)).item()
    print(f"variant={args.variant} batch_size={args.batch_size} input_size={input_size}")
    print(f"replaced={report.replaced_linear_count} skipped={report.skipped_linear_count}")
    print(f"max_abs_err={diff.max().item():.6f}")
    print(f"mean_abs_err={diff.mean().item():.6f}")
    print(f"rmse={rmse:.6f}")
    print(f"cosine_mean={cosine.mean().item():.6f} cosine_min={cosine.min().item():.6f}")
    print(f"top1_agreement={top1_agreement:.6f}")
    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


if __name__ == "__main__":
    main()
