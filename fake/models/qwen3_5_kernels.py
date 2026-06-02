from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fake.kernels.cutlass_nvfp4 import (
    CutlassNVFP4Config,
    ReplacementReport,
    _load_cutlass_nvfp4_symbols,
    replace_linear_with_cutlass_nvfp4,
)
from fake.kernels.cutlass_sparse_bf16 import (
    CutlassSparseBF16Config,
    PaddedSparseBF16Linear,
    SparseBF16ReplacementReport,
    replace_linear_with_cutlass_sparse_bf16,
)
from fake.kernels.cutlass_sparse_nvfp4 import (
    CutlassSparseNVFP4Config,
    PaddedSparseNVFP4Linear,
    SparseReplacementReport,
    replace_linear_with_cutlass_sparse_nvfp4,
)
from fake.kernels.marlin_nvfp4 import (
    MARLIN_CHECKPOINT_FORMAT,
    MarlinNVFP4Config,
    MarlinReplacementReport,
    install_marlin_nvfp4_modules_from_state_dict,
    prepare_marlin_nvfp4_packed_model,
)


QWEN3_5_KERNEL_CHECKPOINT_FORMAT = "qwen3_5_kernel_packed_v1"
QWEN3_5_REAL_KERNEL_METHODS = ("dense", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4")


Report = ReplacementReport | SparseBF16ReplacementReport | SparseReplacementReport | MarlinReplacementReport


@dataclass(frozen=True)
class QwenKernelCheckpointBuildResult:
    metadata: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    report: Report


def prepare_qwen3_5_kernel_checkpoint_payload(
    model: nn.Module,
    *,
    method: str,
    variant: str,
    model_path: str,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> QwenKernelCheckpointBuildResult:
    if method == "dense_nvfp4":
        report = replace_linear_with_cutlass_nvfp4(model, "qwen3_5", CutlassNVFP4Config())
        checkpoint_format = QWEN3_5_KERNEL_CHECKPOINT_FORMAT
    elif method == "sparse_bf16":
        report = replace_linear_with_cutlass_sparse_bf16(
            model,
            "qwen3_5",
            CutlassSparseBF16Config(prune=True),
        )
        checkpoint_format = QWEN3_5_KERNEL_CHECKPOINT_FORMAT
    elif method == "sparse_nvfp4":
        report = replace_linear_with_cutlass_sparse_nvfp4(
            model,
            "qwen3_5",
            CutlassSparseNVFP4Config(prune=True),
        )
        checkpoint_format = QWEN3_5_KERNEL_CHECKPOINT_FORMAT
    elif method == "marlin_nvfp4":
        metadata, report = prepare_marlin_nvfp4_packed_model(
            model,
            "qwen3_5",
            MarlinNVFP4Config(activation_dtype=activation_dtype),
        )
        metadata.update(
            {
                "model_family": "qwen3_5",
                "model_variant": variant,
                "model_path": model_path,
            }
        )
        return QwenKernelCheckpointBuildResult(
            metadata=metadata,
            state_dict=_cpu_state_dict(model),
            report=report,
        )
    else:
        raise ValueError(f"Unsupported Qwen3.5 kernel checkpoint method: {method}")

    metadata = {
        "checkpoint_format": checkpoint_format,
        "method": method,
        "model_family": "qwen3_5",
        "model_variant": variant,
        "model_path": model_path,
        "replacement_backend": report.backend,
        "replacement_config": report.config,
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "skipped": report.skipped,
        "module_specs": _module_specs_from_packed_model(model, method),
    }
    return QwenKernelCheckpointBuildResult(metadata=metadata, state_dict=_cpu_state_dict(model), report=report)


def load_qwen3_5_kernel_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, Any], Report]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid Qwen3.5 kernel checkpoint: {checkpoint_path}")
    metadata = dict(payload.get("metadata", {}))
    state_dict = payload["state_dict"]
    checkpoint_format = metadata.get("checkpoint_format")
    method = metadata.get("method")

    if checkpoint_format == MARLIN_CHECKPOINT_FORMAT or method == "marlin_nvfp4":
        report = install_marlin_nvfp4_modules_from_state_dict(model, state_dict, metadata, device=device)
    elif checkpoint_format == QWEN3_5_KERNEL_CHECKPOINT_FORMAT:
        report = _install_cutlass_modules_from_state_dict(model, state_dict, metadata, device=device)
    else:
        raise ValueError(
            f"Unsupported checkpoint_format={checkpoint_format}; "
            f"expected {QWEN3_5_KERNEL_CHECKPOINT_FORMAT} or {MARLIN_CHECKPOINT_FORMAT}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load Qwen3.5 kernel checkpoint: missing={missing}, unexpected={unexpected}")
    if str(device) != "auto":
        model.to(device)
    model.eval()
    metadata["checkpoint_path"] = str(checkpoint_path)
    metadata["packed_checkpoint_file_size_bytes"] = checkpoint_path.stat().st_size
    return metadata, report


def default_qwen3_5_kernel_checkpoint_path(variant: str, method: str) -> str:
    variant_key = variant.lower().replace(".", "_")
    return f"artifacts/checkpoints/qwen3_5_{variant_key}/{method}/model.pt"


def _install_cutlass_modules_from_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> ReplacementReport | SparseBF16ReplacementReport | SparseReplacementReport:
    method = metadata.get("method")
    if method == "dense_nvfp4":
        return _install_dense_nvfp4(model, state_dict, metadata, device=device)
    if method == "sparse_bf16":
        return _install_sparse_bf16(model, state_dict, metadata, device=device)
    if method == "sparse_nvfp4":
        return _install_sparse_nvfp4(model, state_dict, metadata, device=device)
    raise ValueError(f"Unsupported Qwen3.5 CUTLASS method: {method}")


def _install_dense_nvfp4(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> ReplacementReport:
    wrapper = _load_wrapper()
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        try:
            target_device = _module_target_device(model, name, device)
            weight = wrapper.NVFP4Weight(
                packed_weight=state_dict[f"{name}.packed_weight"].to(target_device),
                scale=state_dict[f"{name}.weight_scale"].to(target_device),
                global_scale=state_dict[f"{name}.weight_global_scale"].to(target_device),
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=_optional_tensor_to(state_dict.get(f"{name}.bias"), target_device),
            )
            _set_module(model, name, wrapper.NVFP4Linear(weight))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return ReplacementReport(
        backend="cutlass_nvfp4_sm120",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )


def _install_sparse_bf16(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> SparseBF16ReplacementReport:
    wrapper = _load_wrapper()
    pad_multiple = int(metadata.get("token_pad_multiple", metadata.get("replacement_config", {}).get("pad_tokens_to_multiple", 8)))
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        prefix = f"{name}.sparse_linear"
        try:
            target_device = _module_target_device(model, name, device)
            weight = wrapper.SparseBF16Weight(
                sparse_weight=state_dict[f"{prefix}.sparse_weight"].to(target_device),
                metadata=state_dict[f"{prefix}.metadata"].to(target_device),
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=_optional_tensor_to(state_dict.get(f"{prefix}.bias"), target_device),
            )
            _set_module(model, name, PaddedSparseBF16Linear(wrapper.SparseBF16Linear(weight), pad_multiple))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return SparseBF16ReplacementReport(
        backend="cutlass_sparse_bf16_cusparselt",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )


def _install_sparse_nvfp4(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> SparseReplacementReport:
    wrapper = _load_wrapper()
    pad_multiple = int(metadata.get("token_pad_multiple", metadata.get("replacement_config", {}).get("pad_tokens_to_multiple", 32)))
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in metadata.get("module_specs", []):
        name = spec["name"]
        prefix = f"{name}.sparse_linear"
        try:
            target_device = _module_target_device(model, name, device)
            weight = wrapper.SparseNVFP4Weight(
                sparse_weight=state_dict[f"{prefix}.sparse_weight"].to(target_device),
                metadata=state_dict[f"{prefix}.metadata"].to(target_device),
                scale=state_dict[f"{prefix}.weight_scale"].to(target_device),
                global_scale=state_dict[f"{prefix}.weight_global_scale"].to(target_device),
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=_optional_tensor_to(state_dict.get(f"{prefix}.bias"), target_device),
            )
            _set_module(model, name, PaddedSparseNVFP4Linear(wrapper.SparseNVFP4Linear(weight), pad_multiple))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return SparseReplacementReport(
        backend="cutlass_sparse_nvfp4_sm120",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )


def _module_specs_from_packed_model(model: nn.Module, method: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    nvfp4_linear_cls, _ = _load_cutlass_nvfp4_symbols()
    for name, module in model.named_modules():
        if method == "dense_nvfp4" and isinstance(module, nvfp4_linear_cls):
            specs.append(_packed_spec(name, module))
        elif method == "sparse_bf16" and isinstance(module, PaddedSparseBF16Linear):
            specs.append(_packed_spec(name, module.sparse_linear))
        elif method == "sparse_nvfp4" and isinstance(module, PaddedSparseNVFP4Linear):
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


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _set_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _get_module(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _module_target_device(
    model: nn.Module,
    module_name: str,
    device: str | torch.device,
) -> torch.device:
    if str(device) != "auto":
        return torch.device(device)
    module = _get_module(model, module_name)
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return tensor.device
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cuda")


def _optional_tensor_to(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if tensor is None else tensor.to(device)


def _load_wrapper():
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


def _dtype_from_string(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return getattr(torch, str(value).replace("torch.", ""), torch.bfloat16)
