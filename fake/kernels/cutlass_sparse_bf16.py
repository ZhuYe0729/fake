from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


SPARSE_BF16_BLOCKED_SHAPES = frozenset({(96, 384)})


@dataclass(frozen=True)
class CutlassSparseBF16Config:
    prune: bool = True
    require_shape_alignment: bool = True
    pad_tokens_to_multiple: int = 8


@dataclass(frozen=True)
class SparseBF16ReplacementReport:
    backend: str
    config: dict[str, Any]
    replaced_linear_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]

    def csv_fields(self) -> dict[str, object]:
        return {
            "kernel_backend": self.backend,
            "sparse_backend": "cusparselt_bf16",
            "sparse_pattern": "2to4",
            "sparse_prune_on_convert": self.config["prune"],
            "token_pad_multiple": self.config["pad_tokens_to_multiple"],
            "replaced_linear_count": self.replaced_linear_count,
            "skipped_linear_count": self.skipped_linear_count,
        }


class PaddedSparseBF16Linear(nn.Module):
    def __init__(self, sparse_linear: nn.Module, pad_multiple: int = 8) -> None:
        super().__init__()
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be positive")
        self.sparse_linear = sparse_linear
        self.pad_multiple = pad_multiple
        self.in_features = int(sparse_linear.in_features)
        self.out_features = int(sparse_linear.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        tokens = int(x_flat.size(0))
        padded_tokens = _round_up(tokens, self.pad_multiple)
        if padded_tokens != tokens:
            x_flat = F.pad(x_flat, (0, 0, 0, padded_tokens - tokens))
        out = self.sparse_linear(x_flat)
        if padded_tokens != tokens:
            out = out[:tokens]
        return out.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"pad_multiple={self.pad_multiple}"
        )


def replace_linear_with_cutlass_sparse_bf16(
    model: nn.Module,
    model_name: str,
    config: CutlassSparseBF16Config | None = None,
) -> SparseBF16ReplacementReport:
    from fake.compression.modules import select_compressible_modules

    config = config or CutlassSparseBF16Config()
    sparse_linear_cls, can_use_sparse = _load_cutlass_sparse_bf16_symbols()
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
        if (linear.out_features, linear.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            skipped.append(
                {
                    "name": module_name,
                    "reason": (
                        "shape_not_supported:cusparselt_sparse_bf16_compressed_size_mismatch:"
                        f"in_features={linear.in_features},out_features={linear.out_features}"
                    ),
                }
            )
            continue
        if config.require_shape_alignment and not can_use_sparse(
            linear.out_features,
            config.pad_tokens_to_multiple,
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
        sparse_linear = sparse_linear_cls.from_linear(linear, prune=config.prune)
        setattr(parent, child_name, PaddedSparseBF16Linear(sparse_linear, config.pad_tokens_to_multiple))
        replaced += 1
    return SparseBF16ReplacementReport(
        backend="cutlass_sparse_bf16_cusparselt",
        config=asdict(config),
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )


def count_cutlass_sparse_bf16_modules(model: nn.Module) -> int:
    sparse_linear_cls, _ = _load_cutlass_sparse_bf16_symbols()
    return sum(1 for module in model.modules() if isinstance(module, sparse_linear_cls))


def cutlass_sparse_bf16_available() -> bool:
    try:
        _load_cutlass_sparse_bf16_symbols()
    except Exception:
        return False
    return True


def _load_cutlass_sparse_bf16_symbols() -> tuple[type[nn.Module], Any]:
    errors: list[str] = []
    for module_name in (
        "fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper",
        "cutlass_wrapper",
    ):
        try:
            module = import_module(module_name)
            return module.SparseBF16Linear, module.can_use_cutlass_sparse_bf16
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "CUTLASS sparse BF16 wrapper package is not importable. "
        "Expected fake/kernels/cutlass/cutlass_wrapper to point at the wrapper repo. "
        f"Tried: {'; '.join(errors)}"
    )


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
