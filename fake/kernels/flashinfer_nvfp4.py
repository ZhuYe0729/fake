from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FlashInferNVFP4Config:
    block_size: int = 16
    sf_layout: str = "layout_128x4"
    gemm_backend: str = "auto"
    quant_backend: str = "cuda"
    out_dtype: str = "auto"
    per_token_activation: bool = False
    fallback_on_error: bool = False
    force_trtllm_weight_shuffle: bool | None = None


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
            "nvfp4_block_size": self.config["block_size"],
            "nvfp4_backend": self.config["gemm_backend"],
            "nvfp4_quant_backend": self.config["quant_backend"],
            "nvfp4_sf_layout": self.config["sf_layout"],
            "nvfp4_per_token_activation": self.config["per_token_activation"],
            "replaced_linear_count": self.replaced_linear_count,
            "skipped_linear_count": self.skipped_linear_count,
            "fallback_count": "",
        }


class FlashInferNVFP4Linear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        config: FlashInferNVFP4Config | None = None,
        name: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.config = config or FlashInferNVFP4Config()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.fallback_calls = 0
        self._flashinfer = _load_flashinfer()
        self._sf_layout = _resolve_sf_layout(self._flashinfer, self.config.sf_layout)
        self._out_dtype = _resolve_out_dtype(self.config.out_dtype, linear.weight.dtype)
        self._weight_shuffle = _weight_needs_shuffle(self.config)

        if self.config.block_size != 16:
            raise ValueError("NVFP4 FlashInfer Linear currently requires block_size=16")
        if self.in_features % self.config.block_size != 0:
            raise ValueError(
                f"in_features={self.in_features} is not divisible by block_size={self.config.block_size}"
            )

        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

        weight = linear.weight.detach().contiguous()
        self.register_buffer("fallback_weight", weight.clone())
        weight_global_sf = _global_scale(weight)
        with torch.inference_mode():
            weight_fp4, weight_sf = self._flashinfer.nvfp4_quantize(
                weight,
                weight_global_sf,
                sfLayout=self._sf_layout,
                do_shuffle=self._weight_shuffle,
                sf_vec_size=self.config.block_size,
                backend=self.config.quant_backend,
            )
        # FlashInfer mm_fp4 expects B as a column-major (K, N) tensor. A plain
        # transpose of the quantized (N, K) weight has the intended stride; making
        # it contiguous would turn it back into row-major layout.
        self.register_buffer("weight_fp4_t", weight_fp4.t())
        self.register_buffer("weight_sf_t", weight_sf.t())
        self.register_buffer("weight_global_sf", weight_global_sf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        try:
            out = self._forward_flashinfer(x_2d)
        except Exception:
            if not self.config.fallback_on_error:
                raise
            self.fallback_calls += 1
            out = F.linear(x_2d, self.fallback_weight, self.bias)
        return out.reshape(*original_shape[:-1], self.out_features)

    def _forward_flashinfer(self, x_2d: torch.Tensor) -> torch.Tensor:
        activation_global_sf = _global_scale(x_2d)
        quant_result = self._flashinfer.nvfp4_quantize(
            x_2d,
            activation_global_sf,
            sfLayout=self._sf_layout,
            do_shuffle=False,
            sf_vec_size=self.config.block_size,
            backend=self.config.quant_backend,
            per_token_activation=self.config.per_token_activation,
        )
        if self.config.per_token_activation:
            activation_fp4, activation_sf, per_token_sf = quant_result
            alpha = per_token_sf.reshape(-1, 1) / self.weight_global_sf
        else:
            activation_fp4, activation_sf = quant_result
            alpha = torch.reciprocal(activation_global_sf * self.weight_global_sf)

        out = self._flashinfer.mm_fp4(
            activation_fp4,
            self.weight_fp4_t,
            activation_sf,
            self.weight_sf_t,
            alpha,
            self._out_dtype,
            None,
            self.config.block_size,
            self.config.sf_layout == "layout_8x4",
            self.config.gemm_backend,
            True,
        )
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return out


def replace_linear_with_flashinfer_nvfp4(
    model: nn.Module,
    model_name: str,
    config: FlashInferNVFP4Config | None = None,
) -> ReplacementReport:
    from fake.compression.modules import select_compressible_modules

    config = config or FlashInferNVFP4Config()
    skipped: list[dict[str, str]] = []
    replaced = 0
    modules = select_compressible_modules(model, model_name)
    for info in modules:
        if info.kind != "linear":
            skipped.append({"name": info.name, "reason": f"unsupported_kind:{info.kind}"})
            continue
        linear = info.module
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": info.name, "reason": f"not_linear:{type(linear).__name__}"})
            continue
        if linear.in_features % config.block_size != 0:
            skipped.append(
                {
                    "name": info.name,
                    "reason": f"in_features_not_divisible_by_{config.block_size}",
                }
            )
            continue
        parent, child_name = _resolve_parent(model, info.name)
        setattr(parent, child_name, FlashInferNVFP4Linear(linear, config=config, name=info.name))
        replaced += 1
    return ReplacementReport(
        backend="flashinfer_nvfp4",
        config=asdict(config),
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
    )


def count_flashinfer_nvfp4_fallbacks(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, FlashInferNVFP4Linear):
            count += module.fallback_calls
    return count


def flashinfer_version() -> str:
    flashinfer = _load_flashinfer()
    return str(getattr(flashinfer, "__version__", "unknown"))


def _load_flashinfer():
    try:
        import flashinfer
    except ImportError as exc:
        raise RuntimeError(
            "FlashInfer is required for the NVFP4 runtime path. "
            "Install/prepare it in the active conda environment before running this benchmark."
        ) from exc

    if hasattr(flashinfer, "nvfp4_quantize") and hasattr(flashinfer, "mm_fp4"):
        return flashinfer

    from flashinfer import gemm
    from flashinfer import fp4_quantization

    class _Namespace:
        nvfp4_quantize = staticmethod(fp4_quantization.nvfp4_quantize)
        mm_fp4 = staticmethod(gemm.mm_fp4)
        SfLayout = flashinfer.SfLayout
        __version__ = getattr(flashinfer, "__version__", "unknown")

    return _Namespace


def _resolve_sf_layout(flashinfer: Any, sf_layout: str) -> Any:
    try:
        return getattr(flashinfer.SfLayout, sf_layout)
    except AttributeError as exc:
        choices = [name for name in dir(flashinfer.SfLayout) if name.startswith("layout_")]
        raise ValueError(f"Unsupported FlashInfer scale layout: {sf_layout}. Choices: {choices}") from exc


def _resolve_out_dtype(out_dtype: str, fallback: torch.dtype) -> torch.dtype:
    if out_dtype == "auto":
        return fallback if fallback in (torch.float16, torch.bfloat16) else torch.bfloat16
    if out_dtype == "fp16":
        return torch.float16
    if out_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported out_dtype: {out_dtype}")


def _weight_needs_shuffle(config: FlashInferNVFP4Config) -> bool:
    if config.force_trtllm_weight_shuffle is not None:
        return config.force_trtllm_weight_shuffle
    return config.gemm_backend == "trtllm"


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    max_abs = x.float().abs().nan_to_num().max().clamp(min=1e-12)
    return torch.tensor([448.0 * 6.0], device=x.device, dtype=torch.float32) / max_abs


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
