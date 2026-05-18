from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fake.compression.checkpoint import load_checkpoint_into_model
from fake.kernels.cutlass_sparse_bf16 import (
    CutlassSparseBF16Config,
    SparseBF16ReplacementReport,
    replace_linear_with_cutlass_sparse_bf16,
)
from fake.models.maxvit import DEFAULT_MAXVIT_VARIANT, load_maxvit_dense


def load_maxvit_cutlass_sparse_bf16(
    model_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    variant: str = DEFAULT_MAXVIT_VARIANT,
    sparse_config: CutlassSparseBF16Config | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], SparseBF16ReplacementReport, dict[str, Any]]:
    model, config = load_maxvit_dense(model_path, dtype="bf16", device=device, variant=variant)
    checkpoint_metadata = load_checkpoint_into_model(model, checkpoint_path)
    model = model.to(dtype=torch.bfloat16)
    report = replace_linear_with_cutlass_sparse_bf16(
        model=model,
        model_name="maxvit",
        config=sparse_config or CutlassSparseBF16Config(),
    )
    model.eval()
    return model, config, report, checkpoint_metadata
