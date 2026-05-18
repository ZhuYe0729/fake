#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fake.models.maxvit import MAXVIT_VARIANT_CHOICES
from fake.models.maxvit_cutlass_checkpoint import (
    default_maxvit_dense_runtime_output,
    default_maxvit_sparse_bf16_runtime_output,
    default_maxvit_sparse_storage_output,
    prepare_maxvit_cutlass_dense_runtime_payload,
    prepare_maxvit_cutlass_sparse_bf16_runtime_payload,
    prepare_maxvit_cutlass_sparse_storage_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MaxViT CUTLASS compressed checkpoints.")
    parser.add_argument("--variant", choices=MAXVIT_VARIANT_CHOICES, required=True)
    parser.add_argument("--backend", choices=["dense_nvfp4", "sparse_nvfp4", "sparse_bf16"], required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None, help="Optional source fake sparse checkpoint for sparse backend.")
    parser.add_argument("--no-prune", action="store_true", help="Sparse only: require source weights to already be sparse.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")
    output = Path(args.output or _default_output(args.variant, args.backend))
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "dense_nvfp4":
        result = prepare_maxvit_cutlass_dense_runtime_payload(
            variant=args.variant,
            model_path=args.model_path,
            output_path=output,
        )
    elif args.backend == "sparse_nvfp4":
        result = prepare_maxvit_cutlass_sparse_storage_payload(
            variant=args.variant,
            model_path=args.model_path,
            output_path=output,
            prune=not args.no_prune,
            checkpoint_path=args.checkpoint,
        )
    else:
        result = prepare_maxvit_cutlass_sparse_bf16_runtime_payload(
            variant=args.variant,
            model_path=args.model_path,
            output_path=output,
            prune=not args.no_prune,
            checkpoint_path=args.checkpoint,
        )
    torch.save({"state_dict": result.state_dict, "metadata": result.metadata}, output)
    with output.with_name("metadata.json").open("w") as f:
        json.dump(result.metadata, f, indent=2)
    print(
        "maxvit cutlass checkpoint prepared: "
        f"variant={args.variant} backend={args.backend} "
        f"format={result.metadata['checkpoint_format']} "
        f"replaced={result.metadata['replaced_linear_count']} "
        f"skipped={result.metadata['skipped_linear_count']} "
        f"output={output} bytes={output.stat().st_size}"
    )


def _default_output(variant: str, backend: str) -> str:
    if backend == "dense_nvfp4":
        return default_maxvit_dense_runtime_output(variant)
    if backend == "sparse_nvfp4":
        return default_maxvit_sparse_storage_output(variant)
    return default_maxvit_sparse_bf16_runtime_output(variant)


if __name__ == "__main__":
    main()
