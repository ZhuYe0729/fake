from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load

from .canonical_nvfp4 import NVFP4CanonicalWeight, canonical_from_linear


_EXT_NAME = "_cutlass_wrapper_nvfp4_sm120"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _remove_unwanted_pytorch_nvcc_flags() -> None:
    for flag in [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]:
        try:
            cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass
    # Remove PyTorch's default arch flags so that only the explicit
    # -gencode=arch=compute_120a,code=sm_120a in extra_cuda_cflags is used.
    # sm_120 does not support block-scaled MMA instructions.
    cpp_extension.COMMON_NVCC_FLAGS[:] = [
        f for f in cpp_extension.COMMON_NVCC_FLAGS
        if not f.startswith("-gencode") and not f.startswith("-arch=")
    ]


@lru_cache(maxsize=1)
def _load_ext() -> Any:
    _remove_unwanted_pytorch_nvcc_flags()
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    root = _repo_root()
    source = root / "cutlass_wrapper" / "csrc" / "nvfp4_sm120.cu"
    build_dir = Path(
        os.environ.get(
            "CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR",
            root / "artifacts" / "torch_extensions",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name=_EXT_NAME,
        sources=[str(source)],
        extra_include_paths=[
            str(root / "cutlass" / "include"),
            str(root / "cutlass" / "tools" / "util" / "include"),
        ],
        extra_cflags=[
            "-O3",
            "-std=c++17",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=os.environ.get("CUTLASS_WRAPPER_NVFP4_VERBOSE_BUILD", "0") == "1",
    )


def _check_2d_cuda_bf16(x: torch.Tensor, name: str) -> None:
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D")
    if x.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be torch.bfloat16")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_supported_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M/N/K must be positive, got M={m}, N={n}, K={k}")
    if k % 32 != 0:
        raise ValueError(f"K must be divisible by 32 for SM120 NVFP4, got K={k}")
    if n % 32 != 0:
        raise ValueError(f"N/out_features must be divisible by 32 for SM120 NVFP4, got N={n}")


def pack_nvfp4_a(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a row-major BF16 activation matrix as CUTLASS NVFP4 A + SFA + global scale."""
    _check_2d_cuda_bf16(x, "x")
    return _load_ext().pack_nvfp4_a(x)


def pack_nvfp4_b(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a row-major BF16 weight matrix [N, K] as CUTLASS NVFP4 B + SFB + global scale."""
    _check_2d_cuda_bf16(weight, "weight")
    return _load_ext().pack_nvfp4_b(weight)


def canonical_to_cutlass_nvfp4_weight(canonical: NVFP4CanonicalWeight) -> "NVFP4Weight":
    """Convert shared logical NVFP4 weight into CUTLASS dense W4A4 storage."""
    _check_supported_shape(1, canonical.out_features, canonical.in_features)
    if canonical.packed_weight.device.type != "cuda":
        raise ValueError("canonical packed weight must be a CUDA tensor")
    if canonical.packed_weight.dtype != torch.uint8:
        raise TypeError("canonical packed weight must be torch.uint8")
    if canonical.packed_weight.shape != (canonical.out_features, canonical.in_features // 2):
        raise ValueError("canonical packed weight shape mismatch")
    if canonical.logical_scale.shape != (canonical.out_features, canonical.in_features // 16):
        raise ValueError("canonical logical scale shape mismatch")
    if canonical.logical_scale.device != canonical.packed_weight.device:
        raise ValueError("canonical logical scale must be on the same CUDA device as packed weight")
    if canonical.global_scale.device != canonical.packed_weight.device:
        raise ValueError("canonical global scale must be on the same CUDA device as packed weight")

    logical_scale_u8 = canonical.logical_scale.contiguous().view(torch.uint8)
    sfb = _load_ext().canonical_scale_to_sfb(
        logical_scale_u8,
        int(canonical.out_features),
        int(canonical.in_features),
    )
    cutlass_global_scale = torch.reciprocal(
        canonical.global_scale.to(device=canonical.packed_weight.device, dtype=torch.float32).reshape(1).contiguous()
    )
    bias = None
    if canonical.bias is not None:
        bias = canonical.bias.to(device=canonical.packed_weight.device, dtype=torch.bfloat16).contiguous()
    return NVFP4Weight(
        packed_weight=canonical.packed_weight.contiguous().view(-1),
        scale=sfb.contiguous(),
        global_scale=cutlass_global_scale,
        in_features=canonical.in_features,
        out_features=canonical.out_features,
        original_dtype=canonical.original_dtype,
        bias=bias,
    )


def can_use_cutlass_nvfp4(m: int, n: int, k: int, *, load_extension: bool = True) -> bool:
    if m <= 0 or n <= 0 or k <= 0 or k % 32 != 0 or n % 32 != 0:
        return False
    if not load_extension:
        return True
    return bool(_load_ext().can_implement_nvfp4_gemm(int(m), int(n), int(k)))


def nvfp4_gemm_bf16(
    a_data: torch.Tensor,
    sfa: torch.Tensor,
    b_data: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | float,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    _check_supported_shape(int(m), int(n), int(k))
    if not torch.is_tensor(alpha):
        alpha = torch.tensor([float(alpha)], device=a_data.device, dtype=torch.float32)
    else:
        alpha = alpha.to(device=a_data.device, dtype=torch.float32).reshape(1).contiguous()
    return _load_ext().nvfp4_gemm_bf16(a_data, sfa, b_data, sfb, alpha, int(m), int(n), int(k))


@dataclass(frozen=True)
class NVFP4Weight:
    packed_weight: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    in_features: int
    out_features: int
    original_dtype: torch.dtype
    bias: torch.Tensor | None = None


@torch.no_grad()
def quantize_weight_bf16(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    original_dtype: torch.dtype | None = None,
) -> NVFP4Weight:
    """Quantize a dense Linear weight [out_features, in_features] into NVFP4 storage."""
    _check_2d_cuda_bf16(weight, "weight")
    from .canonical_nvfp4 import quantize_nvfp4_canonical_weight

    return canonical_to_cutlass_nvfp4_weight(
        quantize_nvfp4_canonical_weight(
            weight.contiguous(),
            bias,
            original_dtype=original_dtype or weight.dtype,
        )
    )


class NVFP4Linear(nn.Module):
    """Inference-only dense NVFP4 replacement for torch.nn.Linear."""

    in_features: int
    out_features: int

    def __init__(self, weight: NVFP4Weight) -> None:
        super().__init__()
        self.in_features = weight.in_features
        self.out_features = weight.out_features
        self.original_dtype = weight.original_dtype
        self.register_buffer("packed_weight", weight.packed_weight.contiguous(), persistent=True)
        self.register_buffer("weight_scale", weight.scale.contiguous(), persistent=True)
        self.register_buffer("weight_global_scale", weight.global_scale.contiguous(), persistent=True)
        if weight.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", weight.bias.contiguous(), persistent=True)
        self.eval()
        self.requires_grad_(False)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, *, device: torch.device | str | None = None) -> "NVFP4Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an instance of torch.nn.Linear")
        target_device = torch.device(device) if device is not None else linear.weight.device
        if target_device.type != "cuda":
            raise ValueError("NVFP4Linear.from_linear requires a CUDA target device")
        weight = linear.weight.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
        return cls(canonical_to_cutlass_nvfp4_weight(canonical_from_linear(linear, device=target_device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("NVFP4Linear is inference-only; call eval() before forward")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("NVFP4Linear does not implement autograd; run under torch.no_grad()")
        if x.device.type != "cuda":
            raise ValueError("x must be a CUDA tensor")
        if x.device != self.packed_weight.device:
            raise ValueError("x must be on the same CUDA device as the NVFP4 weight")
        if x.size(-1) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {x.size(-1)}")

        batch_shape = x.shape[:-1]
        with torch.no_grad():
            x_flat = x.to(dtype=torch.bfloat16).reshape(-1, self.in_features).contiguous()
            m = int(x_flat.size(0))
            _check_supported_shape(m, self.out_features, self.in_features)
            if not can_use_cutlass_nvfp4(m, self.out_features, self.in_features):
                raise RuntimeError(
                    f"CUTLASS NVFP4 GEMM cannot implement M={m}, "
                    f"N={self.out_features}, K={self.in_features}"
                )

            a_packed, a_scale, a_global_scale = pack_nvfp4_a(x_flat)
            alpha = torch.reciprocal(a_global_scale * self.weight_global_scale)
            out = nvfp4_gemm_bf16(
                a_packed,
                a_scale,
                self.packed_weight,
                self.weight_scale,
                alpha,
                m,
                self.out_features,
                self.in_features,
            )
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*batch_shape, self.out_features)
