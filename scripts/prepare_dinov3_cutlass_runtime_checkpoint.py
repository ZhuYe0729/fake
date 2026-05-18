#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config
from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config
from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config
from fake.models.dinov3 import DEFAULT_DINOV3_BACKBONE_PATH, DEFAULT_DINOV3_HEAD_PATH
from fake.models.dinov3_cutlass_nvfp4 import load_dinov3_vit7b16_cutlass_nvfp4_classifier
from fake.models.dinov3_cutlass_runtime import build_runtime_metadata
from fake.models.dinov3_cutlass_sparse_bf16 import load_dinov3_vit7b16_cutlass_sparse_bf16_classifier
from fake.models.dinov3_cutlass_sparse_nvfp4 import load_dinov3_vit7b16_cutlass_sparse_nvfp4_classifier
from fake.models.dinov3_cutlass_storage import sparse_storage_checkpoint_to_runtime_payload


DEFAULT_SPARSE_SOURCE = "artifacts/checkpoints/dinov3_vit7b16/nvfp4_semi_structured_sparse/model.pt"
DEFAULT_SPARSE_BF16_SOURCE = "artifacts/checkpoints/dinov3_vit7b16/semi_structured_sparse/model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DINOv3 CUTLASS runtime-packed checkpoint.")
    parser.add_argument("--backend", choices=["dense_nvfp4", "sparse_nvfp4", "sparse_bf16"], required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--storage-checkpoint", default=None)
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    parser.add_argument("--no-prune", action="store_true", help="Sparse only: require source weights to already be sparse.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    output = Path(args.output or _default_output(args.backend))
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    source_checkpoint = args.source_checkpoint
    source_metadata: dict = {}

    if args.storage_checkpoint:
        if args.backend != "sparse_nvfp4":
            raise ValueError("--storage-checkpoint is currently only supported for --backend sparse_nvfp4")
        payload = sparse_storage_checkpoint_to_runtime_payload(
            args.storage_checkpoint,
            output_path=output,
            device=device,
        )
        torch.save(payload, output)
        metadata = payload["metadata"]
        metadata_path = output.with_name("metadata.json")
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        print(
            "cutlass runtime checkpoint exported from storage: "
            f"backend={args.backend} modules={metadata['replaced_linear_count']} "
            f"skipped={metadata['skipped_linear_count']} output={output} "
            f"bytes={output.stat().st_size} storage={args.storage_checkpoint}"
        )
        return

    if args.backend == "dense_nvfp4":
        model, _config, report = load_dinov3_vit7b16_cutlass_nvfp4_classifier(
            backbone_path=args.backbone_path,
            head_path=args.head_path,
            device=device,
            nvfp4_config=CutlassNVFP4Config(),
        )
    elif args.backend == "sparse_nvfp4":
        if source_checkpoint is None:
            source_checkpoint = DEFAULT_SPARSE_SOURCE
        model, _config, report, source_metadata = load_dinov3_vit7b16_cutlass_sparse_nvfp4_classifier(
            backbone_path=args.backbone_path,
            head_path=args.head_path,
            device=device,
            nvfp4_config=CutlassSparseNVFP4Config(prune=not args.no_prune),
            checkpoint_path=source_checkpoint,
        )
    else:
        if source_checkpoint is None:
            source_checkpoint = DEFAULT_SPARSE_BF16_SOURCE
        model, _config, report, source_metadata = load_dinov3_vit7b16_cutlass_sparse_bf16_classifier(
            backbone_path=args.backbone_path,
            head_path=args.head_path,
            device=device,
            sparse_config=CutlassSparseBF16Config(prune=not args.no_prune),
            checkpoint_path=source_checkpoint,
        )

    metadata = build_runtime_metadata(
        model=model,
        backend=args.backend,
        report=report,
        output_path=output,
        source_checkpoint_path=source_checkpoint,
        source_checkpoint_metadata=source_metadata,
        token_pad_multiple=32 if args.backend == "sparse_nvfp4" else (8 if args.backend == "sparse_bf16" else None),
    )
    payload = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "metadata": metadata}
    torch.save(payload, output)
    metadata_path = output.with_name("metadata.json")
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(
        "cutlass runtime checkpoint exported: "
        f"backend={args.backend} modules={metadata['replaced_linear_count']} "
        f"skipped={metadata['skipped_linear_count']} output={output} "
        f"bytes={output.stat().st_size}"
    )


def _default_output(backend: str) -> str:
    if backend == "dense_nvfp4":
        return "artifacts/checkpoints/dinov3_vit7b16/cutlass_nvfp4_runtime/model.pt"
    if backend == "sparse_nvfp4":
        return "artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_runtime/model.pt"
    return "artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_bf16_runtime/model.pt"


if __name__ == "__main__":
    main()
