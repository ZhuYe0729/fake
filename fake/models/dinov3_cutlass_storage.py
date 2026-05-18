from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fake.kernels.cutlass_sparse_nvfp4 import SparseReplacementReport
from fake.models.dinov3 import DEFAULT_DINOV3_BACKBONE_PATH, DEFAULT_DINOV3_HEAD_PATH
from fake.models.dinov3_cutlass_runtime import (
    RUNTIME_CHECKPOINT_FORMAT,
    RuntimeCheckpointLoadResult,
    _dtype_from_string,
    _load_cutlass_wrapper,
    load_dinov3_vit7b16_cutlass_runtime_payload,
)


STORAGE_CHECKPOINT_FORMAT = "cutlass_storage_packed_v1"
SPARSE_STORAGE_ENCODING = "pairwise_4to8_fp4_pairs_uint8_mask_v1"


@dataclass(frozen=True)
class SparseStorageBuildResult:
    metadata: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


def compact_pairwise_4to8_packed_weight(
    dense_packed_weight: torch.Tensor,
    sparse_weight_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dense_packed_weight.dtype != torch.uint8:
        raise TypeError("dense_packed_weight must be torch.uint8")
    if sparse_weight_bf16.dtype != torch.bfloat16:
        raise TypeError("sparse_weight_bf16 must be torch.bfloat16")
    if dense_packed_weight.dim() != 2 or sparse_weight_bf16.dim() != 2:
        raise ValueError("weight tensors must be 2D")
    rows, half_k = dense_packed_weight.shape
    if sparse_weight_bf16.shape != (rows, half_k * 2):
        raise ValueError("dense_packed_weight and sparse_weight_bf16 shapes do not match")
    if sparse_weight_bf16.size(1) % 8 != 0:
        raise ValueError("in_features must be divisible by 8")

    group_count = int(sparse_weight_bf16.size(1) // 8)
    packed_pairs = dense_packed_weight.reshape(rows, group_count, 4)
    active_pairs = sparse_weight_bf16.reshape(rows, group_count, 4, 2).abs().sum(dim=-1) != 0
    active_count = active_pairs.sum(dim=-1)
    if bool((active_count > 2).any().item()):
        raise ValueError("sparse_weight_bf16 does not satisfy pairwise 4:8 sparsity")

    storage_values = torch.zeros((rows, group_count, 2), device=dense_packed_weight.device, dtype=torch.uint8)
    pair_mask = torch.zeros((rows, group_count), device=dense_packed_weight.device, dtype=torch.uint8)
    for pair_idx in range(4):
        active = active_pairs[:, :, pair_idx]
        rank = active_pairs[:, :, :pair_idx].sum(dim=-1).long()
        if bool(active.any().item()):
            storage_values[:, :, 0].masked_scatter_(
                active & (rank == 0),
                packed_pairs[:, :, pair_idx][active & (rank == 0)],
            )
            storage_values[:, :, 1].masked_scatter_(
                active & (rank == 1),
                packed_pairs[:, :, pair_idx][active & (rank == 1)],
            )
            pair_mask = pair_mask | (active.to(torch.uint8) << pair_idx)
    return storage_values.contiguous(), pair_mask.contiguous()


def unpack_pairwise_4to8_packed_weight(
    storage_values: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    if storage_values.dtype != torch.uint8 or pair_mask.dtype != torch.uint8:
        raise TypeError("storage_values and pair_mask must be torch.uint8")
    if storage_values.dim() != 3 or storage_values.size(-1) != 2:
        raise ValueError("storage_values must have shape [rows, groups, 2]")
    if pair_mask.shape != storage_values.shape[:2]:
        raise ValueError("pair_mask must have shape [rows, groups]")

    rows, groups, _ = storage_values.shape
    packed_pairs = torch.zeros((rows, groups, 4), device=storage_values.device, dtype=torch.uint8)
    for pair_idx in range(4):
        active = ((pair_mask >> pair_idx) & 1).bool()
        rank = torch.zeros_like(pair_mask, dtype=torch.long)
        for lower in range(pair_idx):
            rank = rank + (((pair_mask >> lower) & 1).long())
        if bool(active.any().item()):
            packed_pairs[:, :, pair_idx].masked_scatter_(
                active & (rank == 0),
                storage_values[:, :, 0][active & (rank == 0)],
            )
            packed_pairs[:, :, pair_idx].masked_scatter_(
                active & (rank == 1),
                storage_values[:, :, 1][active & (rank == 1)],
            )
    return packed_pairs.reshape(rows, groups * 4).contiguous()


def build_sparse_storage_checkpoint_payload(
    *,
    model: nn.Module,
    module_specs: list[dict[str, Any]],
    source_checkpoint_path: str | Path | None,
    source_checkpoint_metadata: dict[str, Any] | None,
    output_path: Path,
    report: SparseReplacementReport,
) -> SparseStorageBuildResult:
    wrapper = _load_cutlass_wrapper()
    state_dict: dict[str, torch.Tensor] = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not _is_target_linear_state_key(key, module_specs)
    }

    for spec in module_specs:
        name = spec["name"]
        module = _get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {name}, got {type(module).__name__}")
        weight = module.weight.detach().to(device="cuda", dtype=torch.bfloat16).contiguous()
        wrapper.require_pairwise_4to8_sparse(weight, name)
        packed_weight, scale, global_scale = wrapper.quantize_sparse_nvfp4_bf16(weight)
        storage_values, pair_mask = compact_pairwise_4to8_packed_weight(packed_weight, weight)
        state_dict[f"{name}.storage_values"] = storage_values.detach().cpu()
        state_dict[f"{name}.pair_mask"] = pair_mask.detach().cpu()
        state_dict[f"{name}.weight_scale"] = scale.detach().cpu()
        state_dict[f"{name}.weight_global_scale"] = global_scale.detach().cpu()
        if module.bias is not None:
            state_dict[f"{name}.bias"] = module.bias.detach().to(dtype=torch.bfloat16).cpu().contiguous()

    metadata: dict[str, Any] = {
        "checkpoint_format": STORAGE_CHECKPOINT_FORMAT,
        "backend": "sparse_nvfp4",
        "storage_encoding": SPARSE_STORAGE_ENCODING,
        "storage_checkpoint_path": str(output_path),
        "source_checkpoint_path": str(source_checkpoint_path or ""),
        "source_checkpoint_metadata": source_checkpoint_metadata or {},
        "replacement_backend": report.backend,
        "replacement_config": report.config,
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "skipped": report.skipped,
        "module_specs": module_specs,
        "token_pad_multiple": int(report.config.get("pad_tokens_to_multiple", 32)),
    }
    return SparseStorageBuildResult(metadata=metadata, state_dict=state_dict)


def sparse_storage_checkpoint_to_runtime_payload(
    storage_checkpoint_path: str | Path,
    *,
    output_path: Path,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    checkpoint_path = Path(storage_checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid CUTLASS storage checkpoint: {checkpoint_path}")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("checkpoint_format") != STORAGE_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported checkpoint_format={metadata.get('checkpoint_format')}; "
            f"expected {STORAGE_CHECKPOINT_FORMAT}"
        )
    if metadata.get("backend") != "sparse_nvfp4":
        raise ValueError(f"Unsupported storage backend: {metadata.get('backend')}")

    wrapper = _load_cutlass_wrapper()
    source_state = payload["state_dict"]
    target_device = torch.device(device)
    runtime_state: dict[str, torch.Tensor] = {}
    module_names = {spec["name"] for spec in metadata.get("module_specs", [])}
    for key, value in source_state.items():
        if not torch.is_tensor(value):
            continue
        owner = _storage_key_owner(key)
        if owner in module_names:
            continue
        runtime_state[key] = value

    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        storage_values = source_state[f"{name}.storage_values"].to(device=target_device, non_blocking=True)
        pair_mask = source_state[f"{name}.pair_mask"].to(device=target_device, non_blocking=True)
        dense_packed = unpack_pairwise_4to8_packed_weight(storage_values, pair_mask)
        sparse_weight, sparse_metadata = wrapper.pack_sparse_nvfp4_a_from_dense(dense_packed)
        prefix = f"{name}.sparse_linear"
        runtime_state[f"{prefix}.sparse_weight"] = sparse_weight.detach().cpu()
        runtime_state[f"{prefix}.metadata"] = sparse_metadata.detach().cpu()
        runtime_state[f"{prefix}.weight_scale"] = source_state[f"{name}.weight_scale"]
        runtime_state[f"{prefix}.weight_global_scale"] = source_state[f"{name}.weight_global_scale"]
        if f"{name}.bias" in source_state:
            runtime_state[f"{prefix}.bias"] = source_state[f"{name}.bias"]

    runtime_metadata = {
        **metadata,
        "checkpoint_format": RUNTIME_CHECKPOINT_FORMAT,
        "input_checkpoint_format": STORAGE_CHECKPOINT_FORMAT,
        "runtime_checkpoint_path": str(output_path),
        "storage_checkpoint_path": str(checkpoint_path),
        "storage_checkpoint_format": STORAGE_CHECKPOINT_FORMAT,
        "storage_checkpoint_file_size_bytes": checkpoint_path.stat().st_size,
    }
    return {"state_dict": runtime_state, "metadata": runtime_metadata}


def load_dinov3_vit7b16_cutlass_storage_classifier(
    storage_checkpoint_path: str | Path,
    *,
    backbone_path: str | Path = DEFAULT_DINOV3_BACKBONE_PATH,
    head_path: str | Path = DEFAULT_DINOV3_HEAD_PATH,
    device: str | torch.device = "cuda",
) -> tuple[nn.Module, dict[str, Any], SparseReplacementReport, RuntimeCheckpointLoadResult]:
    checkpoint_path = Path(storage_checkpoint_path)
    payload = sparse_storage_checkpoint_to_runtime_payload(
        checkpoint_path,
        output_path=Path(""),
        device=device,
    )
    metadata = payload["metadata"]
    metadata["runtime_checkpoint_path"] = ""
    model, config, report, load_result = load_dinov3_vit7b16_cutlass_runtime_payload(
        payload,
        checkpoint_path=checkpoint_path,
        file_size_bytes=checkpoint_path.stat().st_size,
        backbone_path=backbone_path,
        head_path=head_path,
        device=device,
        loader_mode="storage_loadtime_pack",
    )
    report = SparseReplacementReport(
        backend=report.backend,
        config=report.config,
        replaced_linear_count=report.replaced_linear_count,
        skipped_linear_count=report.skipped_linear_count,
        skipped=report.skipped,
    )
    return model, config, report, load_result


def sparse_storage_csv_fields(metadata: dict[str, Any] | None) -> dict[str, object]:
    metadata = metadata or {}
    return {
        "storage_checkpoint_path": metadata.get("storage_checkpoint_path", ""),
        "storage_checkpoint_format": metadata.get("storage_checkpoint_format", ""),
        "storage_checkpoint_file_size_bytes": metadata.get("storage_checkpoint_file_size_bytes", ""),
    }


def _is_target_linear_state_key(key: str, module_specs: list[dict[str, Any]]) -> bool:
    return any(key == f"{spec['name']}.weight" or key == f"{spec['name']}.bias" for spec in module_specs)


def _storage_key_owner(key: str) -> str:
    for suffix in ("storage_values", "pair_mask", "weight_scale", "weight_global_scale", "bias"):
        marker = f".{suffix}"
        if key.endswith(marker):
            return key[: -len(marker)]
    return ""


def _get_module(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def module_specs_from_linear_model(model: nn.Module) -> list[dict[str, Any]]:
    from fake.compression.modules import select_compressible_modules

    specs: list[dict[str, Any]] = []
    for info in select_compressible_modules(model, "dinov3_vit7b16"):
        module = info.module
        if isinstance(module, nn.Linear):
            specs.append(
                {
                    "name": info.name,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "bias": module.bias is not None,
                    "original_dtype": str(_dtype_from_string(str(module.weight.dtype))),
                }
            )
    return specs
