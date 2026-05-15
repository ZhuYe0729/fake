from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class CutlassNVFP4Config:
    require_shape_alignment: bool = True


@dataclass(frozen=True)
class ReplacementReport:
    backend: str
    config: dict[str, Any]
    replaced_linear_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]

    def csv_fields(self) -> dict[str, object]:
        return {
            "kernel_backend": self.backend,
            "nvfp4_block_size": 16,
            "nvfp4_backend": "cutlass_sm120",
            "nvfp4_quant_backend": "cutlass_sm120",
            "nvfp4_sf_layout": "cutlass_sm120",
            "replaced_linear_count": self.replaced_linear_count,
            "skipped_linear_count": self.skipped_linear_count,
        }


def replace_linear_with_cutlass_nvfp4(
    model: nn.Module,
    model_name: str,
    config: CutlassNVFP4Config | None = None,
) -> ReplacementReport:
    from fake.compression.modules import select_compressible_modules

    config = config or CutlassNVFP4Config()
    nvfp4_linear_cls, can_use_cutlass_nvfp4 = _load_cutlass_nvfp4_symbols()
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
    return ReplacementReport(
        backend="cutlass_nvfp4_sm120",
        config=asdict(config),
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )


def count_cutlass_nvfp4_modules(model: nn.Module) -> int:
    nvfp4_linear_cls, _ = _load_cutlass_nvfp4_symbols()
    return sum(1 for module in model.modules() if isinstance(module, nvfp4_linear_cls))


def cutlass_nvfp4_available() -> bool:
    try:
        _load_cutlass_nvfp4_symbols()
    except Exception:
        return False
    return True


def _load_cutlass_nvfp4_symbols() -> tuple[type[nn.Module], Any]:
    errors: list[str] = []
    for module_name in (
        "fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper",
        "cutlass_wrapper",
    ):
        try:
            module = import_module(module_name)
            return module.NVFP4Linear, module.can_use_cutlass_nvfp4
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "CUTLASS NVFP4 wrapper package is not importable. "
        "Expected fake/kernels/cutlass/cutlass_wrapper to point at the wrapper repo. "
        f"Tried: {'; '.join(errors)}"
    )


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
