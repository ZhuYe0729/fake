from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fake.compression.modules import select_compressible_modules
from fake.kernels.cutlass_nvfp4 import (
    CutlassNVFP4Config,
    ReplacementReport,
    _load_cutlass_nvfp4_symbols,
    _resolve_parent,
)
from fake.models.maxvit import DEFAULT_MAXVIT_VARIANT, load_maxvit_dense

MAXVIT_DENSE_NVFP4_K_MULTIPLE = 64


def load_maxvit_cutlass_nvfp4(
    model_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    variant: str = DEFAULT_MAXVIT_VARIANT,
    nvfp4_config: CutlassNVFP4Config | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], ReplacementReport]:
    model, config = load_maxvit_dense(model_path, dtype="bf16", device=device, variant=variant)
    model = model.to(dtype=torch.bfloat16)
    report = replace_maxvit_linear_with_cutlass_nvfp4(
        model=model,
        config=nvfp4_config or CutlassNVFP4Config(),
    )
    model.eval()
    return model, config, report


def replace_maxvit_linear_with_cutlass_nvfp4(
    *,
    model: nn.Module,
    config: CutlassNVFP4Config | None = None,
) -> ReplacementReport:
    """Replace only MaxViT Linear shapes that are stable on the dense CUTLASS path."""

    config = config or CutlassNVFP4Config()
    nvfp4_linear_cls, can_use_cutlass_nvfp4 = _load_cutlass_nvfp4_symbols()
    skipped: list[dict[str, str]] = []
    replaced = 0
    selected = select_compressible_modules(model, "maxvit")
    targets = [(info.name, info.kind) for info in selected]
    del selected
    for module_name, kind in targets:
        if kind != "linear":
            skipped.append({"name": module_name, "reason": f"unsupported_kind:{kind}"})
            continue
        parent, child_name = _resolve_parent(model, module_name)
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": module_name, "reason": f"not_linear:{type(linear).__name__}"})
            continue
        if linear.in_features % MAXVIT_DENSE_NVFP4_K_MULTIPLE != 0:
            skipped.append(
                {
                    "name": module_name,
                    "reason": (
                        "shape_not_supported:maxvit_dense_nvfp4_requires_"
                        f"in_features_multiple={MAXVIT_DENSE_NVFP4_K_MULTIPLE},"
                        f"in_features={linear.in_features},out_features={linear.out_features}"
                    ),
                }
            )
            continue
        if config.require_shape_alignment and not can_use_cutlass_nvfp4(
            1,
            linear.out_features,
            linear.in_features,
            load_extension=False,
        ):
            skipped.append(
                {
                    "name": module_name,
                    "reason": (
                        "shape_not_supported:"
                        f"in_features={linear.in_features},out_features={linear.out_features}"
                    ),
                }
            )
            continue
        setattr(parent, child_name, nvfp4_linear_cls.from_linear(linear))
        replaced += 1
    report_config = {
        **asdict(config),
        "maxvit_dense_nvfp4_in_features_multiple": MAXVIT_DENSE_NVFP4_K_MULTIPLE,
    }
    return ReplacementReport(
        backend="cutlass_nvfp4_sm120",
        config=report_config,
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )
