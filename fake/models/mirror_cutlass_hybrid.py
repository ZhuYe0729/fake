from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from fake.compression.modules import select_compressible_modules
from fake.kernels.cutlass_sparse_bf16 import (
    CutlassSparseBF16Config,
    PaddedSparseBF16Linear,
    SPARSE_BF16_BLOCKED_SHAPES,
    _load_cutlass_sparse_bf16_symbols,
)
from fake.kernels.cutlass_sparse_nvfp4 import (
    CutlassSparseNVFP4Config,
    PaddedSparseNVFP4Linear,
    _load_cutlass_sparse_nvfp4_symbols,
)
from fake.models.mirror import (
    DEFAULT_MIRROR_BACKBONE_PATH,
    DEFAULT_MIRROR_MEMORY_PATH,
    DEFAULT_MIRROR_MODEL_PATH,
    load_mirror_dense_detector,
)


HybridScheme = Literal[
    "dino_b16_like",
    "dino_b32_like",
    "attn_nvfp4_mlp_bf16",
    "attn_bf16_mlp_nvfp4",
]

HYBRID_SCHEMES: tuple[str, ...] = (
    "dino_b16_like",
    "dino_b32_like",
    "attn_nvfp4_mlp_bf16",
    "attn_bf16_mlp_nvfp4",
)

ATTN_SUFFIXES = (
    "attention.k_proj.base_layer",
    "attention.v_proj.base_layer",
    "attention.q_proj.base_layer",
    "attention.o_proj",
)
MLP_UP_SUFFIXES = (
    "mlp.gate_proj",
    "mlp.up_proj",
)
MLP_DOWN_SUFFIXES = ("mlp.down_proj",)

SPARSE_NVFP4_SUFFIXES = {
    "dino_b16_like": ATTN_SUFFIXES + MLP_UP_SUFFIXES,
    "dino_b32_like": MLP_UP_SUFFIXES,
    "attn_nvfp4_mlp_bf16": ATTN_SUFFIXES,
    "attn_bf16_mlp_nvfp4": MLP_UP_SUFFIXES + MLP_DOWN_SUFFIXES,
}

SPARSE_BF16_SUFFIXES = {
    "dino_b16_like": MLP_DOWN_SUFFIXES,
    "dino_b32_like": ATTN_SUFFIXES + MLP_DOWN_SUFFIXES,
    "attn_nvfp4_mlp_bf16": MLP_UP_SUFFIXES + MLP_DOWN_SUFFIXES,
    "attn_bf16_mlp_nvfp4": ATTN_SUFFIXES,
}


@dataclass(frozen=True)
class MirrorHybridReplacementReport:
    hybrid_scheme: str
    replaced_linear_count: int
    sparse_nvfp4_module_count: int
    sparse_bf16_module_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]
    sparse_nvfp4_config: dict[str, Any]
    sparse_bf16_config: dict[str, Any]

    def csv_fields(self) -> dict[str, object]:
        return {
            "kernel_backend": "cutlass_hybrid_sparse_nvfp4_sparse_bf16",
            "hybrid_scheme": self.hybrid_scheme,
            "sparse_nvfp4_module_count": self.sparse_nvfp4_module_count,
            "sparse_bf16_module_count": self.sparse_bf16_module_count,
            "sparse_pattern": "pairwise_4to8_and_2to4",
            "sparse_nvfp4_prune_on_convert": self.sparse_nvfp4_config["prune"],
            "sparse_bf16_prune_on_convert": self.sparse_bf16_config["prune"],
            "sparse_nvfp4_token_pad_multiple": self.sparse_nvfp4_config["pad_tokens_to_multiple"],
            "sparse_bf16_token_pad_multiple": self.sparse_bf16_config["pad_tokens_to_multiple"],
            "replaced_linear_count": self.replaced_linear_count,
            "skipped_linear_count": self.skipped_linear_count,
        }


def load_mirror_cutlass_hybrid_detector(
    model_path: str | Path = DEFAULT_MIRROR_MODEL_PATH,
    memory_path: str | Path = DEFAULT_MIRROR_MEMORY_PATH,
    backbone_path: str | Path = DEFAULT_MIRROR_BACKBONE_PATH,
    device: str | torch.device = "cuda",
    hybrid_scheme: HybridScheme = "dino_b32_like",
    sparse_nvfp4_config: CutlassSparseNVFP4Config | None = None,
    sparse_bf16_config: CutlassSparseBF16Config | None = None,
) -> tuple[nn.Module, dict[str, Any], MirrorHybridReplacementReport]:
    model, config = load_mirror_dense_detector(
        model_path=model_path,
        memory_path=memory_path,
        backbone_path=backbone_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    report = replace_mirror_linear_with_cutlass_hybrid(
        model=model,
        hybrid_scheme=hybrid_scheme,
        sparse_nvfp4_config=sparse_nvfp4_config or CutlassSparseNVFP4Config(),
        sparse_bf16_config=sparse_bf16_config or CutlassSparseBF16Config(),
    )
    model.eval()
    return model, config, report


def replace_mirror_linear_with_cutlass_hybrid(
    model: nn.Module,
    hybrid_scheme: HybridScheme,
    sparse_nvfp4_config: CutlassSparseNVFP4Config,
    sparse_bf16_config: CutlassSparseBF16Config,
) -> MirrorHybridReplacementReport:
    if hybrid_scheme not in SPARSE_NVFP4_SUFFIXES:
        raise ValueError(f"Unsupported MIRROR hybrid scheme: {hybrid_scheme}")

    sparse_nvfp4_cls, can_use_sparse_nvfp4 = _load_cutlass_sparse_nvfp4_symbols()
    sparse_bf16_cls, can_use_sparse_bf16 = _load_cutlass_sparse_bf16_symbols()
    skipped: list[dict[str, str]] = []
    sparse_nvfp4_count = 0
    sparse_bf16_count = 0

    selected = select_compressible_modules(model, "mirror")
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

        backend = _backend_for_module(module_name, hybrid_scheme)
        if backend == "sparse_nvfp4":
            if sparse_nvfp4_config.require_shape_alignment and not can_use_sparse_nvfp4(
                linear.out_features,
                sparse_nvfp4_config.pad_tokens_to_multiple,
                linear.in_features,
                load_extension=False,
            ):
                skipped.append({"name": module_name, "reason": _shape_reason(linear, "sparse_nvfp4")})
                continue
            sparse_linear = sparse_nvfp4_cls.from_linear(linear, prune=sparse_nvfp4_config.prune)
            setattr(
                parent,
                child_name,
                PaddedSparseNVFP4Linear(sparse_linear, sparse_nvfp4_config.pad_tokens_to_multiple),
            )
            sparse_nvfp4_count += 1
            continue

        if backend == "sparse_bf16":
            if (linear.out_features, linear.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
                skipped.append({"name": module_name, "reason": _shape_reason(linear, "sparse_bf16_blocked")})
                continue
            if sparse_bf16_config.require_shape_alignment and not can_use_sparse_bf16(
                linear.out_features,
                sparse_bf16_config.pad_tokens_to_multiple,
                linear.in_features,
                load_extension=False,
            ):
                skipped.append({"name": module_name, "reason": _shape_reason(linear, "sparse_bf16")})
                continue
            sparse_linear = sparse_bf16_cls.from_linear(linear, prune=sparse_bf16_config.prune)
            setattr(parent, child_name, PaddedSparseBF16Linear(sparse_linear, sparse_bf16_config.pad_tokens_to_multiple))
            sparse_bf16_count += 1
            continue

        skipped.append({"name": module_name, "reason": f"unsupported_backend:{backend}"})

    return MirrorHybridReplacementReport(
        hybrid_scheme=hybrid_scheme,
        replaced_linear_count=sparse_nvfp4_count + sparse_bf16_count,
        sparse_nvfp4_module_count=sparse_nvfp4_count,
        sparse_bf16_module_count=sparse_bf16_count,
        skipped_linear_count=len(skipped),
        skipped=skipped,
        sparse_nvfp4_config=vars(sparse_nvfp4_config),
        sparse_bf16_config=vars(sparse_bf16_config),
    )


def _backend_for_module(module_name: str, hybrid_scheme: HybridScheme) -> str:
    if module_name.endswith(SPARSE_NVFP4_SUFFIXES[hybrid_scheme]):
        return "sparse_nvfp4"
    if module_name.endswith(SPARSE_BF16_SUFFIXES[hybrid_scheme]):
        return "sparse_bf16"
    return "skip"


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _shape_reason(linear: nn.Linear, backend: str) -> str:
    return f"shape_not_supported:{backend}:in_features={linear.in_features},out_features={linear.out_features}"
