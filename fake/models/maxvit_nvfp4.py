from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fake.kernels.flashinfer_nvfp4 import (
    FlashInferNVFP4Config,
    ReplacementReport,
    replace_linear_with_flashinfer_nvfp4,
)
from fake.models.maxvit import DEFAULT_MAXVIT_VARIANT, load_maxvit_dense


def load_maxvit_flashinfer_nvfp4(
    model_path: str | Path | None = None,
    dtype: str = "bf16",
    device: str | torch.device = "cuda",
    variant: str = DEFAULT_MAXVIT_VARIANT,
    nvfp4_config: FlashInferNVFP4Config | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], ReplacementReport]:
    model, config = load_maxvit_dense(model_path, dtype=dtype, device=device, variant=variant)
    report = replace_linear_with_flashinfer_nvfp4(
        model=model,
        model_name="maxvit",
        config=nvfp4_config or FlashInferNVFP4Config(),
    )
    return model, config, report
