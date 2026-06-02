from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MarlinNVFP4Config:
    activation_dtype: torch.dtype = torch.bfloat16
    require_shape_alignment: bool = True


@dataclass(frozen=True)
class MarlinReplacementReport:
    backend: str
    config: dict[str, Any]
    replaced_linear_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]

    def csv_fields(self) -> dict[str, object]:
        return {
            "kernel_backend": self.backend,
            "nvfp4_block_size": 16,
            "nvfp4_backend": "marlin_sm120",
            "nvfp4_quant_backend": "marlin_sm120",
            "nvfp4_sf_layout": "marlin_sm120",
            "marlin_activation_dtype": str(self.config["activation_dtype"]).replace("torch.", ""),
            "replaced_linear_count": self.replaced_linear_count,
            "skipped_linear_count": self.skipped_linear_count,
        }


MARLIN_CHECKPOINT_FORMAT = "marlin_nvfp4_packed_v1"


def replace_linear_with_marlin_nvfp4(
    model: nn.Module,
    model_name: str,
    config: MarlinNVFP4Config | None = None,
) -> MarlinReplacementReport:
    from fake.compression.modules import select_compressible_modules

    config = config or MarlinNVFP4Config()
    marlin_linear_cls, can_use_marlin = _load_marlin_nvfp4_symbols()
    skipped: list[dict[str, str]] = []
    replaced = 0
    selected = select_compressible_modules(model, model_name)
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
        if config.require_shape_alignment and not can_use_marlin(
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
        setattr(
            parent,
            child_name,
            marlin_linear_cls.from_linear(linear, activation_dtype=config.activation_dtype),
        )
        replaced += 1
    return MarlinReplacementReport(
        backend="marlin_nvfp4_sm120",
        config=_config_dict(config),
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )


def prepare_marlin_nvfp4_packed_model(
    model: nn.Module,
    model_name: str,
    config: MarlinNVFP4Config | None = None,
) -> tuple[dict[str, Any], MarlinReplacementReport]:
    report = replace_linear_with_marlin_nvfp4(model, model_name, config)
    metadata = {
        "checkpoint_format": MARLIN_CHECKPOINT_FORMAT,
        "method": "marlin_nvfp4",
        "model_name": model_name,
        "replacement_backend": report.backend,
        "replacement_config": report.config,
        "activation_dtype": report.config["activation_dtype"],
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "skipped": report.skipped,
        "module_specs": _module_specs_from_marlin_model(model),
    }
    return metadata, report


def load_marlin_nvfp4_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, Any], MarlinReplacementReport]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid Marlin NVFP4 checkpoint: {checkpoint_path}")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("checkpoint_format") != MARLIN_CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported checkpoint_format={metadata.get('checkpoint_format')}; "
            f"expected {MARLIN_CHECKPOINT_FORMAT}"
        )
    state_dict = payload["state_dict"]
    report = install_marlin_nvfp4_modules_from_state_dict(
        model,
        state_dict,
        metadata,
        device=device,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load Marlin NVFP4 checkpoint: missing={missing}, unexpected={unexpected}")
    if str(device) != "auto":
        model.to(device)
    model.eval()
    metadata["checkpoint_path"] = str(checkpoint_path)
    metadata["packed_checkpoint_file_size_bytes"] = checkpoint_path.stat().st_size
    return metadata, report


def install_marlin_nvfp4_modules_from_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> MarlinReplacementReport:
    wrapper = _load_marlin_nvfp4_module()
    activation_dtype = _dtype_from_string(metadata.get("activation_dtype", "torch.bfloat16"))
    specs = list(metadata.get("module_specs", []))
    skipped: list[dict[str, str]] = []
    replaced = 0
    for spec in specs:
        name = spec["name"]
        try:
            target_device = _module_target_device(model, name, device)
            bias = state_dict.get(f"{name}.bias")
            weight = wrapper.MarlinNVFP4Weight(
                packed_weight=state_dict[f"{name}.packed_weight"].to(target_device),
                weight_scale=state_dict[f"{name}.weight_scale"].to(target_device),
                global_scale=state_dict[f"{name}.weight_global_scale"].to(target_device),
                workspace=_make_marlin_workspace(target_device),
                in_features=int(spec["in_features"]),
                out_features=int(spec["out_features"]),
                activation_dtype=activation_dtype,
                original_dtype=_dtype_from_string(spec.get("original_dtype", "torch.bfloat16")),
                bias=bias.to(target_device) if bias is not None else None,
            )
            _set_module(model, name, wrapper.MarlinNVFP4Linear(weight))
            replaced += 1
        except Exception as exc:
            skipped.append({"name": name, "reason": f"{type(exc).__name__}:{exc}"})
    return MarlinReplacementReport(
        backend=str(metadata.get("replacement_backend", "marlin_nvfp4_sm120")),
        config=dict(metadata.get("replacement_config", {})),
        replaced_linear_count=replaced,
        skipped_linear_count=int(metadata.get("skipped_linear_count", 0)) + len(skipped),
        skipped=list(metadata.get("skipped", [])) + skipped,
    )


def count_marlin_nvfp4_modules(model: nn.Module) -> int:
    marlin_linear_cls, _ = _load_marlin_nvfp4_symbols()
    return sum(1 for module in model.modules() if isinstance(module, marlin_linear_cls))


def marlin_nvfp4_available() -> bool:
    try:
        _load_marlin_nvfp4_symbols()
    except Exception:
        return False
    return True


def _load_marlin_nvfp4_symbols() -> tuple[type[nn.Module], Any]:
    errors: list[str] = []
    for module_name in (
        "fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper",
        "cutlass_wrapper",
    ):
        try:
            module = import_module(module_name)
            return module.MarlinNVFP4Linear, module.can_use_marlin_nvfp4
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Marlin NVFP4 wrapper package is not importable. "
        "Expected fake/kernels/cutlass/cutlass_wrapper to point at the wrapper repo. "
        f"Tried: {'; '.join(errors)}"
    )


def _load_marlin_nvfp4_module():
    errors: list[str] = []
    for module_name in (
        "fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper",
        "cutlass_wrapper",
    ):
        try:
            return import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Marlin NVFP4 wrapper package is not importable. Tried: {'; '.join(errors)}")


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _set_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, child_name = _resolve_parent(model, module_name)
    setattr(parent, child_name, new_module)


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


def _module_specs_from_marlin_model(model: nn.Module) -> list[dict[str, Any]]:
    marlin_linear_cls, _ = _load_marlin_nvfp4_symbols()
    specs: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, marlin_linear_cls):
            specs.append(
                {
                    "name": name,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "bias": getattr(module, "bias", None) is not None,
                    "original_dtype": str(getattr(module, "original_dtype", torch.bfloat16)),
                }
            )
    return specs


def _config_dict(config: MarlinNVFP4Config) -> dict[str, Any]:
    data = asdict(config)
    data["activation_dtype"] = str(config.activation_dtype)
    return data


def _dtype_from_string(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return getattr(torch, str(value).replace("torch.", ""), torch.bfloat16)


def _make_marlin_workspace(device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        raise ValueError("Marlin NVFP4 checkpoint loading requires a CUDA device")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms, dtype=torch.int32, device=device)
