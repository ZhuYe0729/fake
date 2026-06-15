from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fake.compression.modules import select_compressible_modules
from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config, _load_cutlass_nvfp4_symbols
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
from fake.models.dinov3 import (
    DEFAULT_DINOV3_BACKBONE_PATH,
    DEFAULT_DINOV3_HEAD_PATH,
    load_dinov3_vit7b16_dense_classifier,
)


DINOV3_LAYERWISE_POLICY_FORMAT = "dinov3_layerwise_speed_policy_v1"
SUPPORTED_BACKENDS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")


@dataclass(frozen=True)
class DINOv3LayerPolicyItem:
    name: str
    backend: str
    n: int
    k: int
    predicted_latency_ms: float | None
    reason: str = ""


@dataclass(frozen=True)
class DINOv3LayerwiseReplacementReport:
    policy_format: str
    policy_path: str
    selected_linear_count: int
    replaced_linear_count: int
    dense_bf16_module_count: int
    dense_nvfp4_module_count: int
    sparse_bf16_module_count: int
    sparse_nvfp4_module_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]
    dense_nvfp4_config: dict[str, Any]
    sparse_bf16_config: dict[str, Any]
    sparse_nvfp4_config: dict[str, Any]

    def csv_fields(self) -> dict[str, object]:
        return {
            "kernel_backend": "dinov3_layerwise_cutlass_policy",
            "policy_format": self.policy_format,
            "policy_path": self.policy_path,
            "selected_linear_count": self.selected_linear_count,
            "replaced_linear_count": self.replaced_linear_count,
            "dense_bf16_module_count": self.dense_bf16_module_count,
            "dense_nvfp4_module_count": self.dense_nvfp4_module_count,
            "sparse_bf16_module_count": self.sparse_bf16_module_count,
            "sparse_nvfp4_module_count": self.sparse_nvfp4_module_count,
            "skipped_linear_count": self.skipped_linear_count,
            "dense_nvfp4_require_shape_alignment": self.dense_nvfp4_config["require_shape_alignment"],
            "sparse_bf16_prune_on_convert": self.sparse_bf16_config["prune"],
            "sparse_bf16_token_pad_multiple": self.sparse_bf16_config["pad_tokens_to_multiple"],
            "sparse_nvfp4_prune_on_convert": self.sparse_nvfp4_config["prune"],
            "sparse_nvfp4_token_pad_multiple": self.sparse_nvfp4_config["pad_tokens_to_multiple"],
        }


def load_dinov3_vit7b16_layerwise_policy_classifier(
    policy_path: str | Path,
    backbone_path: str | Path = DEFAULT_DINOV3_BACKBONE_PATH,
    head_path: str | Path = DEFAULT_DINOV3_HEAD_PATH,
    device: str | torch.device = "cuda",
    dense_nvfp4_config: CutlassNVFP4Config | None = None,
    sparse_bf16_config: CutlassSparseBF16Config | None = None,
    sparse_nvfp4_config: CutlassSparseNVFP4Config | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], DINOv3LayerwiseReplacementReport]:
    model, config = load_dinov3_vit7b16_dense_classifier(
        backbone_path=backbone_path,
        head_path=head_path,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(dtype=torch.bfloat16)
    report = replace_dinov3_linear_with_layerwise_policy(
        model=model,
        policy_path=policy_path,
        dense_nvfp4_config=dense_nvfp4_config or CutlassNVFP4Config(),
        sparse_bf16_config=sparse_bf16_config or CutlassSparseBF16Config(),
        sparse_nvfp4_config=sparse_nvfp4_config or CutlassSparseNVFP4Config(),
    )
    model.eval()
    return model, config, report


def replace_dinov3_linear_with_layerwise_policy(
    model: nn.Module,
    policy_path: str | Path,
    dense_nvfp4_config: CutlassNVFP4Config,
    sparse_bf16_config: CutlassSparseBF16Config,
    sparse_nvfp4_config: CutlassSparseNVFP4Config,
) -> DINOv3LayerwiseReplacementReport:
    policy_file = Path(policy_path)
    policy_items = load_dinov3_layer_policy(policy_file)
    policy_by_name = {item.name: item for item in policy_items}
    selected = select_compressible_modules(model, "dinov3_vit7b16")
    selected_names = {info.name for info in selected if info.kind == "linear"}

    unknown_policy_names = sorted(set(policy_by_name) - selected_names)
    skipped = [{"name": name, "reason": "policy_name_not_compressible"} for name in unknown_policy_names]
    backend_counts: Counter[str] = Counter()
    replaced = 0

    dense_nvfp4_cls, can_use_dense_nvfp4 = _load_cutlass_nvfp4_symbols()
    sparse_bf16_cls, can_use_sparse_bf16 = _load_cutlass_sparse_bf16_symbols()
    sparse_nvfp4_cls, can_use_sparse_nvfp4 = _load_cutlass_sparse_nvfp4_symbols()

    for info in selected:
        if info.kind != "linear":
            skipped.append({"name": info.name, "reason": f"unsupported_kind:{info.kind}"})
            continue
        item = policy_by_name.get(info.name)
        if item is None:
            skipped.append({"name": info.name, "reason": "missing_policy"})
            continue
        parent, child_name = _resolve_parent(model, info.name)
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": info.name, "reason": f"not_linear:{type(linear).__name__}"})
            continue
        if item.backend not in SUPPORTED_BACKENDS:
            skipped.append({"name": info.name, "reason": f"unsupported_backend:{item.backend}"})
            continue

        if item.backend == "dense_bf16":
            backend_counts[item.backend] += 1
            continue

        if item.backend == "dense_nvfp4":
            if dense_nvfp4_config.require_shape_alignment and not can_use_dense_nvfp4(
                1,
                linear.out_features,
                linear.in_features,
                load_extension=False,
            ):
                skipped.append({"name": info.name, "reason": _shape_reason(linear, item.backend)})
                continue
            setattr(parent, child_name, dense_nvfp4_cls.from_linear(linear))
            backend_counts[item.backend] += 1
            replaced += 1
            continue

        if item.backend == "sparse_bf16":
            if (linear.out_features, linear.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
                skipped.append({"name": info.name, "reason": _shape_reason(linear, "sparse_bf16_blocked")})
                continue
            if sparse_bf16_config.require_shape_alignment and not can_use_sparse_bf16(
                linear.out_features,
                sparse_bf16_config.pad_tokens_to_multiple,
                linear.in_features,
                load_extension=False,
            ):
                skipped.append({"name": info.name, "reason": _shape_reason(linear, item.backend)})
                continue
            sparse_linear = sparse_bf16_cls.from_linear(linear, prune=sparse_bf16_config.prune)
            setattr(parent, child_name, PaddedSparseBF16Linear(sparse_linear, sparse_bf16_config.pad_tokens_to_multiple))
            backend_counts[item.backend] += 1
            replaced += 1
            continue

        if item.backend == "sparse_nvfp4":
            if sparse_nvfp4_config.require_shape_alignment and not can_use_sparse_nvfp4(
                linear.out_features,
                sparse_nvfp4_config.pad_tokens_to_multiple,
                linear.in_features,
                load_extension=False,
            ):
                skipped.append({"name": info.name, "reason": _shape_reason(linear, item.backend)})
                continue
            sparse_linear = sparse_nvfp4_cls.from_linear(linear, prune=sparse_nvfp4_config.prune)
            setattr(parent, child_name, PaddedSparseNVFP4Linear(sparse_linear, sparse_nvfp4_config.pad_tokens_to_multiple))
            backend_counts[item.backend] += 1
            replaced += 1

    return DINOv3LayerwiseReplacementReport(
        policy_format=DINOV3_LAYERWISE_POLICY_FORMAT,
        policy_path=str(policy_file),
        selected_linear_count=len(policy_items),
        replaced_linear_count=replaced,
        dense_bf16_module_count=backend_counts["dense_bf16"],
        dense_nvfp4_module_count=backend_counts["dense_nvfp4"],
        sparse_bf16_module_count=backend_counts["sparse_bf16"],
        sparse_nvfp4_module_count=backend_counts["sparse_nvfp4"],
        skipped_linear_count=len(skipped),
        skipped=skipped,
        dense_nvfp4_config=asdict(dense_nvfp4_config),
        sparse_bf16_config=asdict(sparse_bf16_config),
        sparse_nvfp4_config=asdict(sparse_nvfp4_config),
    )


def load_dinov3_layer_policy(path: str | Path) -> list[DINOv3LayerPolicyItem]:
    payload = json.loads(Path(path).read_text())
    if payload.get("policy_format") != DINOV3_LAYERWISE_POLICY_FORMAT:
        raise ValueError(f"Unsupported DINOv3 layerwise policy format: {payload.get('policy_format')}")
    return [
        DINOv3LayerPolicyItem(
            name=str(row["name"]),
            backend=str(row["backend"]),
            n=int(row["n"]),
            k=int(row["k"]),
            predicted_latency_ms=_optional_float(row.get("predicted_latency_ms")),
            reason=str(row.get("reason", "")),
        )
        for row in payload.get("modules", [])
    ]


def policy_item_to_dict(item: DINOv3LayerPolicyItem) -> dict[str, Any]:
    return asdict(item)


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _shape_reason(linear: nn.Linear, backend: str) -> str:
    return f"shape_not_supported:{backend}:in_features={linear.in_features},out_features={linear.out_features}"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
