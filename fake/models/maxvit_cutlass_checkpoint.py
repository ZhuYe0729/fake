from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fake.compression.modules import select_compressible_modules
from fake.kernels.cutlass_nvfp4 import ReplacementReport
from fake.kernels.cutlass_sparse_bf16 import (
    CutlassSparseBF16Config,
    PaddedSparseBF16Linear,
    SPARSE_BF16_BLOCKED_SHAPES,
    SparseBF16ReplacementReport,
)
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear, SparseReplacementReport
from fake.models.dinov3_cutlass_runtime import RUNTIME_CHECKPOINT_FORMAT, RuntimeCheckpointLoadResult
from fake.models.dinov3_cutlass_storage import (
    SPARSE_STORAGE_ENCODING,
    STORAGE_CHECKPOINT_FORMAT,
    compact_pairwise_4to8_packed_weight,
    unpack_pairwise_4to8_packed_weight,
)
from fake.models.maxvit import DEFAULT_MAXVIT_VARIANT, get_maxvit_variant, load_maxvit_dense
from fake.models.maxvit_cutlass_nvfp4 import MAXVIT_DENSE_NVFP4_K_MULTIPLE


@dataclass(frozen=True)
class MaxViTCheckpointBuildResult:
    metadata: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


def prepare_maxvit_cutlass_dense_runtime_payload(
    *,
    variant: str,
    model_path: str | Path | None,
    output_path: Path,
) -> MaxViTCheckpointBuildResult:
    from fake.models.maxvit_cutlass_nvfp4 import load_maxvit_cutlass_nvfp4

    model, config, report = load_maxvit_cutlass_nvfp4(model_path=model_path, device="cuda", variant=variant)
    metadata = {
        "checkpoint_format": RUNTIME_CHECKPOINT_FORMAT,
        "backend": "dense_nvfp4",
        "model_family": "maxvit",
        "model_variant": variant,
        "model_id": config.get("model_id", ""),
        "runtime_checkpoint_path": str(output_path),
        "replacement_backend": report.backend,
        "replacement_config": report.config,
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "skipped": report.skipped,
        "module_specs": _module_specs_from_packed_model(model, "dense_nvfp4"),
    }
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return MaxViTCheckpointBuildResult(metadata=metadata, state_dict=state_dict)


def prepare_maxvit_cutlass_sparse_storage_payload(
    *,
    variant: str,
    model_path: str | Path | None,
    output_path: Path,
    prune: bool = True,
    checkpoint_path: str | Path | None = None,
) -> MaxViTCheckpointBuildResult:
    from fake.compression.checkpoint import load_checkpoint_into_model

    wrapper = _load_cutlass_wrapper()
    model, config = load_maxvit_dense(model_path=model_path, dtype="bf16", device="cuda", variant=variant)
    source_metadata = load_checkpoint_into_model(model, checkpoint_path)
    specs, skipped = _select_supported_maxvit_linear_specs(model, backend="sparse_nvfp4")
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not _is_target_linear_state_key(key, specs)
    }
    for spec in specs:
        name = spec["name"]
        module = _get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {name}, got {type(module).__name__}")
        weight = module.weight.detach().to(device="cuda", dtype=torch.bfloat16).contiguous()
        sparse_weight = wrapper.magnitude_prune_pairwise_4to8_bf16(weight).contiguous() if prune else weight
        if not prune:
            wrapper.require_pairwise_4to8_sparse(sparse_weight, name)
        packed_weight, scale, global_scale = wrapper.quantize_sparse_nvfp4_bf16(sparse_weight)
        storage_values, pair_mask = compact_pairwise_4to8_packed_weight(packed_weight, sparse_weight)
        state_dict[f"{name}.storage_values"] = storage_values.detach().cpu()
        state_dict[f"{name}.pair_mask"] = pair_mask.detach().cpu()
        state_dict[f"{name}.weight_scale"] = scale.detach().cpu()
        state_dict[f"{name}.weight_global_scale"] = global_scale.detach().cpu()
        if module.bias is not None:
            state_dict[f"{name}.bias"] = module.bias.detach().to(dtype=torch.bfloat16).cpu().contiguous()

    report = SparseReplacementReport(
        backend="cutlass_sparse_nvfp4_sm120",
        config={"prune": prune, "require_shape_alignment": True, "pad_tokens_to_multiple": 32},
        replaced_linear_count=len(specs),
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )
    metadata = {
        "checkpoint_format": STORAGE_CHECKPOINT_FORMAT,
        "backend": "sparse_nvfp4",
        "model_family": "maxvit",
        "model_variant": variant,
        "model_id": config.get("model_id", ""),
        "storage_encoding": SPARSE_STORAGE_ENCODING,
        "storage_checkpoint_path": str(output_path),
        "source_checkpoint_path": str(checkpoint_path or ""),
        "source_checkpoint_metadata": source_metadata,
        "replacement_backend": report.backend,
        "replacement_config": report.config,
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "skipped": report.skipped,
        "module_specs": specs,
        "token_pad_multiple": 32,
    }
    return MaxViTCheckpointBuildResult(metadata=metadata, state_dict=state_dict)


def prepare_maxvit_cutlass_sparse_bf16_runtime_payload(
    *,
    variant: str,
    model_path: str | Path | None,
    output_path: Path,
    prune: bool = True,
    checkpoint_path: str | Path | None = None,
) -> MaxViTCheckpointBuildResult:
    from fake.models.maxvit_cutlass_sparse_bf16 import load_maxvit_cutlass_sparse_bf16

    model, config, report, source_metadata = load_maxvit_cutlass_sparse_bf16(
        model_path=model_path,
        device="cuda",
        variant=variant,
        sparse_config=CutlassSparseBF16Config(prune=prune),
        checkpoint_path=checkpoint_path,
    )
    metadata = {
        "checkpoint_format": RUNTIME_CHECKPOINT_FORMAT,
        "backend": "sparse_bf16",
        "model_family": "maxvit",
        "model_variant": variant,
        "model_id": config.get("model_id", ""),
        "runtime_checkpoint_path": str(output_path),
        "source_checkpoint_path": str(checkpoint_path or ""),
        "source_checkpoint_metadata": source_metadata,
        "replacement_backend": report.backend,
        "replacement_config": report.config,
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "skipped": report.skipped,
        "module_specs": _module_specs_from_packed_model(model, "sparse_bf16"),
        "token_pad_multiple": 8,
    }
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return MaxViTCheckpointBuildResult(metadata=metadata, state_dict=state_dict)


def load_maxvit_cutlass_dense_runtime(
    runtime_checkpoint_path: str | Path,
    *,
    model_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    variant: str = DEFAULT_MAXVIT_VARIANT,
) -> tuple[nn.Module, dict[str, Any], ReplacementReport, RuntimeCheckpointLoadResult]:
    checkpoint_path = Path(runtime_checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata, state_dict = _validate_payload(payload, RUNTIME_CHECKPOINT_FORMAT, "dense_nvfp4", checkpoint_path)
    model, config = load_maxvit_dense(model_path=model_path, dtype="bf16", device="cpu", variant=variant)
    wrapper = _load_cutlass_wrapper()
    skipped: list[dict[str, str]] = []
    replaced = 0
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        try:
            weight = wrapper.NVFP4Weight(
                packed_weight=state_dict[f"{name}.packed_weight"],
                scale=state_dict[f"{name}.weight_scale"],
                global_scale=state_dict[f"{name}.weight_global_scale"],
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=state_dict.get(f"{name}.bias"),
            )
            _set_module(model, name, wrapper.NVFP4Linear(weight))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load MaxViT dense runtime checkpoint: missing={missing}, unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    report = ReplacementReport(
        backend="cutlass_nvfp4_sm120",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )
    load_result = RuntimeCheckpointLoadResult(
        metadata={**metadata, "runtime_checkpoint_path": str(checkpoint_path)},
        file_size_bytes=checkpoint_path.stat().st_size,
        loader_mode="maxvit_runtime_assign",
    )
    return model, config, report, load_result


def load_maxvit_cutlass_sparse_bf16_runtime(
    runtime_checkpoint_path: str | Path,
    *,
    model_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    variant: str = DEFAULT_MAXVIT_VARIANT,
) -> tuple[nn.Module, dict[str, Any], SparseBF16ReplacementReport, RuntimeCheckpointLoadResult]:
    checkpoint_path = Path(runtime_checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata, state_dict = _validate_payload(payload, RUNTIME_CHECKPOINT_FORMAT, "sparse_bf16", checkpoint_path)
    model, config = load_maxvit_dense(model_path=model_path, dtype="bf16", device="cpu", variant=variant)
    wrapper = _load_cutlass_wrapper()
    skipped: list[dict[str, str]] = []
    replaced = 0
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        prefix = f"{name}.sparse_linear"
        try:
            weight = wrapper.SparseBF16Weight(
                sparse_weight=state_dict[f"{prefix}.sparse_weight"],
                metadata=state_dict[f"{prefix}.metadata"],
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=state_dict.get(f"{prefix}.bias"),
            )
            sparse_linear = wrapper.SparseBF16Linear(weight)
            _set_module(model, name, PaddedSparseBF16Linear(sparse_linear, int(metadata.get("token_pad_multiple", 8))))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load MaxViT sparse BF16 runtime checkpoint: missing={missing}, unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    report = SparseBF16ReplacementReport(
        backend="cutlass_sparse_bf16_cusparselt",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )
    load_result = RuntimeCheckpointLoadResult(
        metadata={**metadata, "runtime_checkpoint_path": str(checkpoint_path)},
        file_size_bytes=checkpoint_path.stat().st_size,
        loader_mode="maxvit_runtime_assign",
    )
    return model, config, report, load_result


def load_maxvit_cutlass_sparse_storage(
    storage_checkpoint_path: str | Path,
    *,
    model_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    variant: str = DEFAULT_MAXVIT_VARIANT,
) -> tuple[nn.Module, dict[str, Any], SparseReplacementReport, RuntimeCheckpointLoadResult]:
    checkpoint_path = Path(storage_checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata, source_state = _validate_payload(payload, STORAGE_CHECKPOINT_FORMAT, "sparse_nvfp4", checkpoint_path)
    wrapper = _load_cutlass_wrapper()
    model, config = load_maxvit_dense(model_path=model_path, dtype="bf16", device="cpu", variant=variant)
    runtime_state: dict[str, torch.Tensor] = {}
    module_names = {spec["name"] for spec in metadata.get("module_specs", [])}
    for key, value in source_state.items():
        if not torch.is_tensor(value):
            continue
        owner = _storage_key_owner(key)
        if owner in module_names:
            continue
        runtime_state[key] = value

    replaced = 0
    skipped: list[dict[str, str]] = []
    target_device = torch.device(device)
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        try:
            storage_values = source_state[f"{name}.storage_values"].to(device=target_device)
            pair_mask = source_state[f"{name}.pair_mask"].to(device=target_device)
            dense_packed = unpack_pairwise_4to8_packed_weight(storage_values, pair_mask)
            sparse_weight, sparse_metadata = wrapper.pack_sparse_nvfp4_a_from_dense(dense_packed)
            weight = wrapper.SparseNVFP4Weight(
                sparse_weight=sparse_weight.detach().cpu(),
                metadata=sparse_metadata.detach().cpu(),
                scale=source_state[f"{name}.weight_scale"],
                global_scale=source_state[f"{name}.weight_global_scale"],
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=source_state.get(f"{name}.bias"),
            )
            sparse_linear = wrapper.SparseNVFP4Linear(weight)
            _set_module(model, name, PaddedSparseNVFP4Linear(sparse_linear, int(metadata.get("token_pad_multiple", 32))))
            prefix = f"{name}.sparse_linear"
            runtime_state[f"{prefix}.sparse_weight"] = weight.sparse_weight
            runtime_state[f"{prefix}.metadata"] = weight.metadata
            runtime_state[f"{prefix}.weight_scale"] = weight.scale
            runtime_state[f"{prefix}.weight_global_scale"] = weight.global_scale
            if weight.bias is not None:
                runtime_state[f"{prefix}.bias"] = weight.bias
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    missing, unexpected = model.load_state_dict(runtime_state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load MaxViT sparse storage checkpoint: missing={missing}, unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    report = SparseReplacementReport(
        backend="cutlass_sparse_nvfp4_sm120",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )
    load_result = RuntimeCheckpointLoadResult(
        metadata={
            **metadata,
            "input_checkpoint_format": STORAGE_CHECKPOINT_FORMAT,
            "runtime_checkpoint_path": "",
            "storage_checkpoint_path": str(checkpoint_path),
            "storage_checkpoint_format": STORAGE_CHECKPOINT_FORMAT,
            "storage_checkpoint_file_size_bytes": checkpoint_path.stat().st_size,
        },
        file_size_bytes=checkpoint_path.stat().st_size,
        loader_mode="storage_loadtime_pack",
    )
    return model, config, report, load_result


def _select_supported_maxvit_linear_specs(model: nn.Module, backend: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    wrapper = _load_cutlass_wrapper()
    specs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for info in select_compressible_modules(model, "maxvit"):
        if info.kind != "linear":
            skipped.append({"name": info.name, "reason": f"unsupported_kind:{info.kind}"})
            continue
        module = info.module
        if not isinstance(module, nn.Linear):
            skipped.append({"name": info.name, "reason": f"not_linear:{type(module).__name__}"})
            continue
        if backend == "sparse_nvfp4":
            supported = wrapper.can_use_cutlass_sparse_nvfp4(module.out_features, 32, module.in_features)
        elif backend == "sparse_bf16":
            supported = wrapper.can_use_cutlass_sparse_bf16(
                module.out_features,
                8,
                module.in_features,
                load_extension=False,
            ) and (module.out_features, module.in_features) not in SPARSE_BF16_BLOCKED_SHAPES
        else:
            supported = (
                module.in_features % MAXVIT_DENSE_NVFP4_K_MULTIPLE == 0
                and wrapper.can_use_cutlass_nvfp4(1, module.out_features, module.in_features, load_extension=False)
            )
        if not supported:
            skipped.append(
                {
                    "name": info.name,
                    "reason": f"shape_not_supported:in_features={module.in_features},out_features={module.out_features}",
                }
            )
            continue
        specs.append(_linear_spec(info.name, module))
    return specs, skipped


def _module_specs_from_packed_model(model: nn.Module, backend: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if backend == "dense_nvfp4" and hasattr(module, "packed_weight"):
            specs.append(_packed_spec(name, module))
        elif backend == "sparse_bf16" and isinstance(module, PaddedSparseBF16Linear):
            specs.append(_packed_spec(name, module.sparse_linear))
    return specs


def _packed_spec(name: str, module: nn.Module) -> dict[str, Any]:
    return {
        "name": name,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "bias": getattr(module, "bias", None) is not None,
        "original_dtype": str(getattr(module, "original_dtype", torch.bfloat16)),
    }


def _linear_spec(name: str, module: nn.Linear) -> dict[str, Any]:
    return {
        "name": name,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "bias": module.bias is not None,
        "original_dtype": str(module.weight.dtype),
    }


def _validate_payload(
    payload: dict[str, Any],
    checkpoint_format: str,
    backend: str,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid MaxViT CUTLASS checkpoint: {checkpoint_path}")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("checkpoint_format") != checkpoint_format:
        raise ValueError(f"Unsupported checkpoint_format={metadata.get('checkpoint_format')}; expected {checkpoint_format}")
    if metadata.get("backend") != backend:
        raise ValueError(f"Unsupported backend={metadata.get('backend')}; expected {backend}")
    return metadata, payload["state_dict"]


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


def _set_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _dtype_from_string(value: str) -> torch.dtype:
    return getattr(torch, value.replace("torch.", ""), torch.bfloat16)


def _load_cutlass_wrapper():
    import importlib

    for module_name in (
        "fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper",
        "cutlass_wrapper",
    ):
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass
    raise RuntimeError("CUTLASS wrapper package is not importable")


def default_maxvit_dense_runtime_output(variant: str) -> str:
    return f"artifacts/checkpoints/maxvit_{variant}/cutlass_nvfp4_runtime/model.pt"


def default_maxvit_sparse_storage_output(variant: str) -> str:
    return f"artifacts/checkpoints/maxvit_{variant}/cutlass_sparse_nvfp4_storage/model.pt"


def default_maxvit_sparse_bf16_runtime_output(variant: str) -> str:
    return f"artifacts/checkpoints/maxvit_{variant}/cutlass_sparse_bf16_runtime/model.pt"
