from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load


_EXT_NAME = "official_marlin_int4_ext_vllm"
_DEFAULT_ARCH_LIST = "8.0;8.6;8.9;9.0;12.0"
_DEFAULT_GROUPSIZE = 128
_MAX_PAR = 16
_QZERO = 8
_MAXQ = 15
_PACKED_ZERO_8 = -2004318072  # int32 representation of 0x88888888


def _sanitize_arch_list(value: str | None) -> str:
    if not value:
        return _DEFAULT_ARCH_LIST

    kept: list[str] = []
    for raw_arch in value.replace(",", ";").split(";"):
        arch = raw_arch.strip()
        if not arch:
            continue
        arch_num = arch.split("+", 1)[0]
        try:
            major = int(arch_num.split(".", 1)[0])
        except ValueError:
            continue
        if major >= 8:
            kept.append(arch)
    return ";".join(kept) if kept else _DEFAULT_ARCH_LIST


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vllm_marlin_root() -> Path:
    return _repo_root() / "third_party" / "vllm"


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


@lru_cache(maxsize=1)
def _load_ext() -> Any:
    _remove_unwanted_pytorch_nvcc_flags()
    # Official Marlin uses cp.async and requires SM80+; user-provided lists on
    # shared environments often include older cards that would break ptxas.
    os.environ["TORCH_CUDA_ARCH_LIST"] = _sanitize_arch_list(os.environ.get("TORCH_CUDA_ARCH_LIST"))

    root = _repo_root()
    binding_dir = _repo_root() / "standalone_marlin_nvfp4" / "csrc"
    source_dir = _vllm_marlin_root() / "csrc"
    marlin_dir = source_dir / "quantization" / "marlin"
    build_dir = Path(
        os.environ.get(
            "MARLIN_INT4_EXT_BUILD_DIR",
            root / "artifacts" / "torch_extensions_marlin_int4",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    kernel_sources = sorted(str(path) for path in marlin_dir.glob("sm*_kernel_*.cu"))

    return load(
        name=_EXT_NAME,
        sources=[
            str(binding_dir / "torch_bindings.cpp"),
            str(marlin_dir / "marlin.cu"),
            str(marlin_dir / "gptq_marlin_repack.cu"),
            *kernel_sources,
        ],
        extra_include_paths=[str(source_dir)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-static-global-template-stub=false",
        ],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=os.environ.get("MARLIN_INT4_VERBOSE_BUILD", "0") == "1",
    )


def _scalar_type_id(
    exponent: int,
    mantissa: int,
    signed: bool,
    bias: int,
    finite_values_only: bool = False,
    nan_repr: int = 1,
) -> int:
    value = 0
    offset = 0

    def add(member: int, bit_width: int) -> None:
        nonlocal value
        nonlocal offset
        value |= (int(member) & ((1 << bit_width) - 1)) << offset
        offset += bit_width

    add(exponent, 8)
    add(mantissa, 8)
    add(signed, 1)
    add(bias, 32)
    add(finite_values_only, 1)
    add(nan_repr, 8)
    return value


UINT4_ID = _scalar_type_id(exponent=0, mantissa=4, signed=False, bias=0)
UINT4_BIAS8_ID = _scalar_type_id(exponent=0, mantissa=4, signed=False, bias=8)


@lru_cache(maxsize=1)
def _scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])

    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])

    return scale_perm, scale_perm_single


def _check_supported_shape(m: int, n: int, k: int, groupsize: int = _DEFAULT_GROUPSIZE) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M/N/K must be positive, got M={m}, N={n}, K={k}")
    if groupsize not in (-1, _DEFAULT_GROUPSIZE):
        raise ValueError("Marlin INT4 only supports groupsize=-1 or groupsize=128")
    if k % 128 != 0:
        raise ValueError(f"K/in_features must be divisible by 128 for Marlin INT4, got K={k}")
    if n % 256 != 0:
        raise ValueError(f"N/out_features must be divisible by 256 for Marlin INT4, got N={n}")
    if groupsize != -1 and k % groupsize != 0:
        raise ValueError(f"K/in_features must be divisible by groupsize={groupsize}, got K={k}")


def can_use_marlin_int4(
    m: int,
    n: int,
    k: int,
    *,
    groupsize: int = _DEFAULT_GROUPSIZE,
    load_extension: bool = False,
) -> bool:
    try:
        _check_supported_shape(int(m), int(n), int(k), int(groupsize))
    except ValueError:
        return False
    if load_extension:
        try:
            _load_ext()
        except Exception:
            return False
    return True


def _check_2d_cuda_float(weight: torch.Tensor, name: str) -> None:
    if weight.dim() != 2:
        raise ValueError(f"{name} must be 2D")
    if weight.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be float16, bfloat16, or float32")
    if not weight.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _normalize_groupsize(groupsize: int, k: int) -> int:
    if groupsize not in (-1, _DEFAULT_GROUPSIZE):
        raise ValueError("Marlin INT4 only supports groupsize=-1 or groupsize=128")
    return k if groupsize == -1 else groupsize


def _quantize_weight_to_marlin_format(
    weight: torch.Tensor,
    groupsize: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = int(weight.size(0)), int(weight.size(1))
    effective_groupsize = _normalize_groupsize(groupsize, k)
    scale_perm, scale_perm_single = _scale_perms()

    w = torch.nan_to_num(weight.to(torch.float16).T.contiguous(), nan=0.0, posinf=0.0, neginf=0.0)
    if effective_groupsize != k:
        grouped = w.reshape((-1, effective_groupsize, n)).permute(1, 0, 2).reshape(effective_groupsize, -1)
    else:
        grouped = w

    scales_flat = (grouped.abs().amax(dim=0, keepdim=True) * (2.0 / _MAXQ)).clamp_min(1.0e-8)
    q_grouped = torch.clamp(torch.round(grouped / scales_flat) + _QZERO, 0, _MAXQ).to(torch.int32)
    ref_grouped = (q_grouped - _QZERO).to(torch.float16) * scales_flat.to(torch.float16)

    if effective_groupsize != k:
        q_w = q_grouped.reshape((effective_groupsize, -1, n)).permute(1, 0, 2).reshape(k, n).contiguous()
        ref = ref_grouped.reshape((effective_groupsize, -1, n)).permute(1, 0, 2).reshape(k, n).contiguous()
        scales = scales_flat.reshape((-1, n)).contiguous()
        marlin_scales = scales_flat.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        q_w = q_grouped.reshape(k, n).contiguous()
        ref = ref_grouped.reshape(k, n).contiguous()
        scales = scales_flat.reshape((-1, n)).contiguous()
        marlin_scales = scales_flat.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]

    marlin_scales = marlin_scales.reshape((-1, n)).contiguous().to(torch.float16)
    low = q_w.T.contiguous()[:, 0::2].to(torch.uint8)
    high = (q_w.T.contiguous()[:, 1::2].to(torch.uint8) << 4)
    canonical_packed = (low | high).contiguous()
    return canonical_packed, marlin_scales, ref.T.contiguous()


@dataclass(frozen=True)
class MarlinInt4Weight:
    packed_weight: torch.Tensor
    weight_scale: torch.Tensor
    zero_points: torch.Tensor | None
    workspace: torch.Tensor
    in_features: int
    out_features: int
    groupsize: int
    original_dtype: torch.dtype
    bias: torch.Tensor | None = None


def _make_workspace(device: torch.device | str | int, n: int) -> torch.Tensor:
    if isinstance(device, int):
        device = torch.device("cuda", device)
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Marlin INT4 workspace must be allocated on a CUDA device")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(max(n // 128 * _MAX_PAR, sms), dtype=torch.int32, device=device)


def _make_zero_points(device: torch.device, k: int, n: int, groupsize: int) -> torch.Tensor | None:
    if groupsize == -1:
        return None
    num_groups = k // groupsize
    packed_zero = torch.full(
        (num_groups, n // 8),
        _PACKED_ZERO_8,
        dtype=torch.int32,
        device=device,
    )
    return packed_zero.contiguous()


@torch.no_grad()
def quantize_marlin_int4_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    groupsize: int = _DEFAULT_GROUPSIZE,
    original_dtype: torch.dtype | None = None,
) -> MarlinInt4Weight:
    """Quantize dense Linear weight [out_features, in_features] into official Marlin INT4 storage."""
    _check_2d_cuda_float(weight, "weight")
    n, k = int(weight.size(0)), int(weight.size(1))
    _check_supported_shape(1, n, k, groupsize)
    _load_ext()
    canonical_packed, scale, _ = _quantize_weight_to_marlin_format(weight, groupsize)
    perm = torch.empty(0, dtype=torch.int32, device=weight.device)
    qweight = canonical_packed.contiguous().view(torch.int32).T.contiguous()
    packed = torch.ops.official_marlin_int4_ext_vllm.gptq_marlin_repack(
        qweight,
        perm,
        k,
        n,
        4,
        False,
    )

    stored_bias = None
    if bias is not None:
        if bias.dim() != 1 or bias.numel() != n:
            raise ValueError(f"bias must have shape ({n},)")
        if bias.device != weight.device:
            raise ValueError("bias must be on the same device as weight")
        stored_bias = bias.to(device=weight.device, dtype=torch.float16).contiguous()

    return MarlinInt4Weight(
        packed_weight=packed.contiguous(),
        weight_scale=scale.contiguous(),
        zero_points=_make_zero_points(weight.device, k, n, groupsize),
        workspace=_make_workspace(weight.device, n),
        in_features=k,
        out_features=n,
        groupsize=groupsize,
        original_dtype=original_dtype or weight.dtype,
        bias=stored_bias,
    )


def marlin_int4_gemm(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    zero_points: torch.Tensor | None,
    workspace: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    _check_supported_shape(int(m), int(n), int(k), -1 if weight_scale.size(0) == 1 else int(k // weight_scale.size(0)))
    if x.dtype != torch.float16:
        raise TypeError("x must be torch.float16")
    _load_ext()
    b_type_id = UINT4_BIAS8_ID if zero_points is None else UINT4_ID
    return torch.ops.official_marlin_int4_ext_vllm.marlin_gemm(
        x,
        None,
        packed_weight,
        None,
        weight_scale,
        None,
        None,
        zero_points,
        None,
        None,
        workspace,
        b_type_id,
        int(m),
        int(n),
        int(k),
        True,
        False,
        True,
        False,
    )


def simulate_marlin_int4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    groupsize: int = _DEFAULT_GROUPSIZE,
) -> torch.Tensor:
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("simulate_marlin_int4_linear expects CUDA tensors")
    if x.size(-1) != weight.size(1):
        raise ValueError(f"expected input last dim {weight.size(1)}, got {x.size(-1)}")
    _check_supported_shape(1, int(weight.size(0)), int(weight.size(1)), groupsize)
    _, _, dequant_weight = _quantize_weight_to_marlin_format(weight.contiguous(), groupsize)
    bias_cast = bias.to(device=x.device, dtype=torch.float16) if bias is not None else None
    out = F.linear(x.to(torch.float16), dequant_weight.to(torch.float16), bias_cast)
    return out.to(torch.float16)


class MarlinInt4Linear(nn.Module):
    """Inference-only official Marlin FP16xINT4 W4A16 replacement for torch.nn.Linear."""

    in_features: int
    out_features: int

    def __init__(self, weight: MarlinInt4Weight) -> None:
        super().__init__()
        self.in_features = weight.in_features
        self.out_features = weight.out_features
        self.groupsize = weight.groupsize
        self.original_dtype = weight.original_dtype
        self.register_buffer("packed_weight", weight.packed_weight.contiguous(), persistent=True)
        self.register_buffer("weight_scale", weight.weight_scale.contiguous(), persistent=True)
        if weight.zero_points is None:
            self.register_buffer("zero_points", None, persistent=True)
        else:
            self.register_buffer("zero_points", weight.zero_points.contiguous(), persistent=True)
        self.register_buffer("workspace", weight.workspace.contiguous(), persistent=False)
        if weight.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", weight.bias.contiguous(), persistent=True)
        self.eval()
        self.requires_grad_(False)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        device: torch.device | str | None = None,
        groupsize: int = _DEFAULT_GROUPSIZE,
    ) -> "MarlinInt4Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an instance of torch.nn.Linear")
        target_device = torch.device(device) if device is not None else linear.weight.device
        if target_device.type != "cuda":
            raise ValueError("MarlinInt4Linear.from_linear requires a CUDA target device")
        weight = linear.weight.detach().to(device=target_device, dtype=torch.float16).contiguous()
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(device=target_device, dtype=torch.float16).contiguous()
        return cls(
            quantize_marlin_int4_weight(
                weight,
                bias,
                groupsize=groupsize,
                original_dtype=linear.weight.dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("MarlinInt4Linear is inference-only; call eval() before forward")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("MarlinInt4Linear does not implement autograd; run under torch.no_grad()")
        if x.device.type != "cuda":
            raise ValueError("x must be a CUDA tensor")
        if x.device != self.packed_weight.device:
            raise ValueError("x must be on the same CUDA device as the Marlin INT4 weight")
        if x.size(-1) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {x.size(-1)}")

        batch_shape = x.shape[:-1]
        with torch.no_grad():
            x_flat = x.to(dtype=torch.float16).reshape(-1, self.in_features).contiguous()
            m = int(x_flat.size(0))
            _check_supported_shape(m, self.out_features, self.in_features, self.groupsize)
            out = marlin_int4_gemm(
                x_flat,
                self.packed_weight,
                self.weight_scale,
                self.zero_points,
                self.workspace,
                m,
                self.out_features,
                self.in_features,
            )
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*batch_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, groupsize={self.groupsize}"
        )


__all__ = [
    "MarlinInt4Linear",
    "MarlinInt4Weight",
    "UINT4_ID",
    "UINT4_BIAS8_ID",
    "can_use_marlin_int4",
    "marlin_int4_gemm",
    "quantize_marlin_int4_weight",
    "simulate_marlin_int4_linear",
]
