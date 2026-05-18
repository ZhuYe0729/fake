#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from fake.compression.checkpoint import load_checkpoint_into_model
from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config, SparseReplacementReport
from fake.models.dinov3 import (
    DEFAULT_DINOV3_BACKBONE_PATH,
    DEFAULT_DINOV3_HEAD_PATH,
    load_dinov3_vit7b16_dense_classifier,
)
from fake.models.dinov3_cutlass_storage import (
    build_sparse_storage_checkpoint_payload,
    module_specs_from_linear_model,
)


DEFAULT_SPARSE_SOURCE = "artifacts/checkpoints/dinov3_vit7b16/nvfp4_semi_structured_sparse/model.pt"
DEFAULT_OUTPUT = "artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DINOv3 CUTLASS sparse NVFP4 storage checkpoint.")
    parser.add_argument("--backend", choices=["sparse_nvfp4"], default="sparse_nvfp4")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SPARSE_SOURCE)
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, _config = load_dinov3_vit7b16_dense_classifier(
        backbone_path=args.backbone_path,
        head_path=args.head_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    source_metadata = load_checkpoint_into_model(model, args.source_checkpoint)
    model = model.to(dtype=torch.bfloat16)
    specs = module_specs_from_linear_model(model)
    report = SparseReplacementReport(
        backend="cutlass_sparse_nvfp4_storage",
        config=asdict(CutlassSparseNVFP4Config(prune=False)),
        replaced_linear_count=len(specs),
        skipped_linear_count=0,
        skipped=[],
    )
    result = build_sparse_storage_checkpoint_payload(
        model=model,
        module_specs=specs,
        source_checkpoint_path=args.source_checkpoint,
        source_checkpoint_metadata=source_metadata,
        output_path=output,
        report=report,
    )
    torch.save({"state_dict": result.state_dict, "metadata": result.metadata}, output)
    metadata_path = output.with_name("metadata.json")
    with metadata_path.open("w") as f:
        json.dump(result.metadata, f, indent=2)
    print(
        "cutlass storage checkpoint exported: "
        f"backend={args.backend} modules={result.metadata['replaced_linear_count']} "
        f"skipped={result.metadata['skipped_linear_count']} output={output} "
        f"bytes={output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
