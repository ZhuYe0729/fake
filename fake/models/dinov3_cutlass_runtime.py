from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fake.compression.modules import select_compressible_modules
from fake.kernels.cutlass_nvfp4 import ReplacementReport
from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear, SparseBF16ReplacementReport
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear, SparseReplacementReport
from fake.models.dinov3 import (
    DEFAULT_DINOV3_BACKBONE_PATH,
    DEFAULT_DINOV3_HEAD_PATH,
    DINOv3LinearClassifier,
    _load_config,
)


RUNTIME_CHECKPOINT_FORMAT = "cutlass_runtime_packed_v1"


@dataclass(frozen=True)
class RuntimeCheckpointLoadResult:
    metadata: dict[str, Any]
    file_size_bytes: int
    loader_mode: str = "meta_skeleton_assign"

    def csv_fields(self) -> dict[str, object]:
        return {
            "checkpoint_format": self.metadata.get("input_checkpoint_format", self.metadata.get("checkpoint_format", "")),
            "runtime_checkpoint_path": self.metadata.get("runtime_checkpoint_path", ""),
            "source_checkpoint_path": self.metadata.get("source_checkpoint_path", ""),
            "packed_checkpoint_file_size_bytes": self.file_size_bytes if self.metadata.get("runtime_checkpoint_path") else "",
            "runtime_checkpoint_loader_mode": self.loader_mode,
            "storage_checkpoint_path": self.metadata.get("storage_checkpoint_path", ""),
            "storage_checkpoint_format": self.metadata.get("storage_checkpoint_format", ""),
            "storage_checkpoint_file_size_bytes": self.metadata.get("storage_checkpoint_file_size_bytes", ""),
        }


def load_dinov3_vit7b16_cutlass_runtime_classifier(
    runtime_checkpoint_path: str | Path,
    backbone_path: str | Path = DEFAULT_DINOV3_BACKBONE_PATH,
    head_path: str | Path = DEFAULT_DINOV3_HEAD_PATH,
    device: str | torch.device = "cuda",
) -> tuple[
    nn.Module,
    dict[str, Any],
    ReplacementReport | SparseReplacementReport | SparseBF16ReplacementReport,
    RuntimeCheckpointLoadResult,
]:
    checkpoint_path = Path(runtime_checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    return load_dinov3_vit7b16_cutlass_runtime_payload(
        payload,
        checkpoint_path=checkpoint_path,
        file_size_bytes=checkpoint_path.stat().st_size,
        backbone_path=backbone_path,
        head_path=head_path,
        device=device,
        loader_mode="meta_skeleton_assign",
    )


def load_dinov3_vit7b16_cutlass_runtime_payload(
    payload: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    file_size_bytes: int,
    backbone_path: str | Path = DEFAULT_DINOV3_BACKBONE_PATH,
    head_path: str | Path = DEFAULT_DINOV3_HEAD_PATH,
    device: str | torch.device = "cuda",
    loader_mode: str = "meta_skeleton_assign",
) -> tuple[
    nn.Module,
    dict[str, Any],
    ReplacementReport | SparseReplacementReport | SparseBF16ReplacementReport,
    RuntimeCheckpointLoadResult,
]:
    checkpoint_path = Path(checkpoint_path)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid CUTLASS runtime checkpoint: {checkpoint_path}")
    state_dict = payload["state_dict"]
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("checkpoint_format") != RUNTIME_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported checkpoint_format={metadata.get('checkpoint_format')}; "
            f"expected {RUNTIME_CHECKPOINT_FORMAT}"
        )

    config = _load_config(Path(backbone_path))
    model = _build_meta_dinov3_classifier(backbone_path, head_path, config)
    backend = metadata.get("backend")
    if backend == "dense_nvfp4":
        report = _install_dense_nvfp4_modules(model, state_dict, metadata)
    elif backend == "sparse_nvfp4":
        report = _install_sparse_nvfp4_modules(model, state_dict, metadata)
    elif backend == "sparse_bf16":
        report = _install_sparse_bf16_modules(model, state_dict, metadata)
    else:
        raise ValueError(f"Unsupported CUTLASS runtime backend: {backend}")

    missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load runtime checkpoint cleanly: missing={missing}, unexpected={unexpected}")
    _materialize_dinov3_nonpersistent_buffers(model, config)
    model = model.to(device)
    model.eval()
    runtime_checkpoint_path = str(checkpoint_path) if loader_mode != "storage_loadtime_pack" else metadata.get(
        "runtime_checkpoint_path", ""
    )
    load_result = RuntimeCheckpointLoadResult(
        metadata={**metadata, "runtime_checkpoint_path": runtime_checkpoint_path},
        file_size_bytes=file_size_bytes,
        loader_mode=loader_mode,
    )
    return model, config, report, load_result


def runtime_checkpoint_csv_fields(load_result: RuntimeCheckpointLoadResult | None) -> dict[str, object]:
    if load_result is None:
        return {
            "checkpoint_format": "",
            "runtime_checkpoint_path": "",
            "source_checkpoint_path": "",
            "packed_checkpoint_file_size_bytes": "",
            "runtime_checkpoint_loader_mode": "",
            "storage_checkpoint_path": "",
            "storage_checkpoint_format": "",
            "storage_checkpoint_file_size_bytes": "",
        }
    return load_result.csv_fields()


def _build_meta_dinov3_classifier(backbone_path: str | Path, head_path: str | Path, config: dict[str, Any]) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    backbone_dir = Path(backbone_path)
    hf_config = AutoConfig.from_pretrained(str(backbone_dir), local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        backbone = AutoModel.from_config(hf_config, trust_remote_code=False)
        head = nn.Linear(2 * int(config["hidden_size"]), 1000)
    return DINOv3LinearClassifier(
        backbone=backbone,
        linear_head=head,
        num_register_tokens=int(config.get("num_register_tokens", 4)),
    )


def _materialize_dinov3_nonpersistent_buffers(model: nn.Module, config: dict[str, Any]) -> None:
    rope = getattr(getattr(model, "backbone", None), "rope_embeddings", None)
    inv_freq = getattr(rope, "inv_freq", None)
    if rope is None or inv_freq is None or not getattr(inv_freq, "is_meta", False):
        return
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    head_dim = hidden_size // num_attention_heads
    materialized = 1 / float(config.get("rope_theta", 100.0)) ** torch.arange(
        0,
        1,
        4 / head_dim,
        dtype=torch.float32,
    )
    rope.register_buffer("inv_freq", materialized, persistent=False)


def _install_dense_nvfp4_modules(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> ReplacementReport:
    wrapper = _load_cutlass_wrapper()
    specs = metadata.get("module_specs") or _module_specs_from_model(model)
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in specs:
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
    return ReplacementReport(
        backend="cutlass_nvfp4_sm120",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )


def _install_sparse_nvfp4_modules(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> SparseReplacementReport:
    wrapper = _load_cutlass_wrapper()
    specs = metadata.get("module_specs") or _module_specs_from_model(model)
    pad_multiple = int(metadata.get("token_pad_multiple", 32))
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in specs:
        name = spec["name"]
        prefix = f"{name}.sparse_linear"
        try:
            weight = wrapper.SparseNVFP4Weight(
                sparse_weight=state_dict[f"{prefix}.sparse_weight"],
                metadata=state_dict[f"{prefix}.metadata"],
                scale=state_dict[f"{prefix}.weight_scale"],
                global_scale=state_dict[f"{prefix}.weight_global_scale"],
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=state_dict.get(f"{prefix}.bias"),
            )
            _set_module(model, name, PaddedSparseNVFP4Linear(wrapper.SparseNVFP4Linear(weight), pad_multiple))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return SparseReplacementReport(
        backend="cutlass_sparse_nvfp4_sm120",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )


def _install_sparse_bf16_modules(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> SparseBF16ReplacementReport:
    wrapper = _load_cutlass_wrapper()
    specs = metadata.get("module_specs") or _module_specs_from_model(model)
    pad_multiple = int(metadata.get("token_pad_multiple", 8))
    replaced = 0
    skipped: list[dict[str, str]] = []
    for spec in specs:
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
            _set_module(model, name, PaddedSparseBF16Linear(wrapper.SparseBF16Linear(weight), pad_multiple))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return SparseBF16ReplacementReport(
        backend="cutlass_sparse_bf16_cusparselt",
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )


def build_runtime_metadata(
    *,
    model: nn.Module,
    backend: str,
    report: ReplacementReport | SparseReplacementReport | SparseBF16ReplacementReport,
    output_path: Path,
    source_checkpoint_path: str | Path | None,
    source_checkpoint_metadata: dict[str, Any] | None = None,
    token_pad_multiple: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "checkpoint_format": RUNTIME_CHECKPOINT_FORMAT,
        "backend": backend,
        "runtime_checkpoint_path": str(output_path),
        "source_checkpoint_path": str(source_checkpoint_path or ""),
        "source_checkpoint_metadata": source_checkpoint_metadata or {},
        "replacement_backend": report.backend,
        "replacement_config": report.config,
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "skipped": report.skipped,
        "module_specs": _module_specs_from_packed_model(model, backend),
    }
    if token_pad_multiple is not None:
        metadata["token_pad_multiple"] = token_pad_multiple
    return metadata


def _module_specs_from_model(model: nn.Module) -> list[dict[str, Any]]:
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
                    "original_dtype": str(module.weight.dtype),
                }
            )
    return specs


def _module_specs_from_packed_model(model: nn.Module, backend: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if backend == "dense_nvfp4" and hasattr(module, "packed_weight"):
            specs.append(_packed_spec(name, module))
        elif backend == "sparse_nvfp4" and isinstance(module, PaddedSparseNVFP4Linear):
            specs.append(_packed_spec(name, module.sparse_linear))
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


def _set_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, child_name = _resolve_parent(model, module_name)
    setattr(parent, child_name, new_module)


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _dtype_from_string(value: str) -> torch.dtype:
    name = value.replace("torch.", "")
    return getattr(torch, name, torch.bfloat16)
