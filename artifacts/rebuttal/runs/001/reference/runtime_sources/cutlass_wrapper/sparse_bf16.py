from __future__ import annotations

import importlib.util
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


_EXT_NAME = "_cutlass_wrapper_sparse_bf16_backends"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cusparselt_root() -> Path:
    override = os.environ.get("CUSPARSELT_ROOT")
    if override:
        return Path(override)
    spec = importlib.util.find_spec("nvidia.cusparselt")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            "Unable to locate nvidia.cusparselt. Activate the wja-cospaq environment "
            "or set CUSPARSELT_ROOT to a directory containing include/ and lib/."
        )
    return Path(next(iter(spec.submodule_search_locations)))


def cusparselt_paths() -> tuple[Path, Path]:
    root = _cusparselt_root()
    include_dir = root / "include"
    lib_path = root / "lib" / "libcusparseLt.so.0"
    if not (include_dir / "cusparseLt.h").exists():
        raise RuntimeError(f"cuSPARSELt header not found under {include_dir}")
    if not lib_path.exists():
        raise RuntimeError(f"cuSPARSELt library not found at {lib_path}")
    return include_dir, lib_path


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


def _ensure_cuda_home() -> None:
    _ensure_supported_cuda_arch_list()
    if os.environ.get("CUDA_HOME"):
        return
    for candidate in (Path("/data/apps/cuda/12.8"), Path("/usr/local/cuda")):
        if (candidate / "include" / "cuda_runtime.h").exists():
            os.environ["CUDA_HOME"] = str(candidate)
            cpp_extension.CUDA_HOME = str(candidate)
            return


def _ensure_supported_cuda_arch_list() -> None:
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
        return

    kept = []
    for raw_arch in arch_list.replace(";", " ").split():
        arch = raw_arch.removesuffix("+PTX")
        try:
            major = int(arch.split(".", maxsplit=1)[0])
        except ValueError:
            continue
        if major >= 8:
            kept.append(raw_arch)
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(kept) if kept else "12.0"


@lru_cache(maxsize=1)
def _load_ext() -> Any:
    _remove_unwanted_pytorch_nvcc_flags()
    _ensure_cuda_home()
    root = _repo_root()
    source_dir = root / "cutlass_wrapper" / "csrc"
    spinfer_dir = root / "spinfer_standalone"
    include_dir, lib_path = cusparselt_paths()
    build_dir = Path(
        os.environ.get(
            "CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR",
            root / "artifacts" / "torch_extensions_sparse_bf16_backends",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name=_EXT_NAME,
        sources=[str(source_dir / "sparse_bf16_bindings.cpp"), str(spinfer_dir / "src" / "spinfer_api.cu")],
        extra_include_paths=[str(include_dir), str(spinfer_dir / "include"), str(spinfer_dir / "src")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-maxrregcount=255", "--use_fast_math"],
        extra_ldflags=[str(lib_path), f"-Wl,-rpath,{lib_path.parent}", "-lcublas", "-lcusparse"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=os.environ.get("CUTLASS_WRAPPER_SPARSE_BF16_VERBOSE_BUILD", "0") == "1",
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


def _check_sparse_bf16_supported_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M/N/K must be positive, got M={m}, N={n}, K={k}")
    if m % 8 != 0:
        raise ValueError(f"M/out_features must be divisible by 8 for cuSPARSELt sparse BF16, got M={m}")
    if k % 64 != 0:
        raise ValueError(f"K/in_features must be divisible by 64 for cuSPARSELt sparse BF16, got K={k}")
    if n % 8 != 0:
        raise ValueError(f"N/tokens must be divisible by 8 for cuSPARSELt sparse BF16 v1, got N={n}")


def _check_spinfer_sparse_bf16_supported_shape(m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"M/N/K must be positive, got M={m}, N={n}, K={k}")
    if m % 64 != 0:
        raise ValueError(f"M/out_features must be divisible by 64 for SpInfer sparse BF16, got M={m}")
    if k % 64 != 0:
        raise ValueError(f"K/in_features must be divisible by 64 for SpInfer sparse BF16, got K={k}")
    if n not in (8, 16, 32, 64, 128) and n % 128 != 0:
        raise ValueError(
            "N/tokens must be one of 8,16,32,64,128 or divisible by 128 "
            f"for SpInfer sparse BF16, got N={n}"
        )


def _normalize_sparse_bf16_backend(backend: str) -> str:
    normalized = backend.lower()
    if normalized not in ("cusparselt", "spinfer"):
        raise ValueError(f"unknown sparse BF16 backend {backend!r}; expected 'cusparselt' or 'spinfer'")
    return normalized


def can_use_cusparselt_sparse_bf16(m: int, n: int, k: int, *, load_extension: bool = False) -> bool:
    if not (m > 0 and n > 0 and k > 0 and m % 8 == 0 and n % 8 == 0 and k % 64 == 0):
        return False
    if load_extension:
        try:
            _load_ext()
        except Exception:
            return False
    return True


def can_use_cutlass_sparse_bf16(m: int, n: int, k: int, *, load_extension: bool = False) -> bool:
    return can_use_cusparselt_sparse_bf16(m, n, k, load_extension=load_extension)


def can_use_spinfer_sparse_bf16(m: int, n: int, k: int, *, load_extension: bool = False) -> bool:
    if not (
        m > 0
        and n > 0
        and k > 0
        and m % 64 == 0
        and k % 64 == 0
        and (n in (8, 16, 32, 64, 128) or n % 128 == 0)
    ):
        return False
    if load_extension:
        try:
            _load_ext()
        except Exception:
            return False
    return True


def is_2to4_sparse(weight: torch.Tensor) -> bool:
    if weight.numel() == 0 or weight.size(-1) % 4 != 0:
        return False
    groups = weight.reshape(*weight.shape[:-1], weight.shape[-1] // 4, 4)
    return bool((groups != 0).sum(dim=-1).le(2).all().item())


def require_2to4_sparse(weight: torch.Tensor, name: str = "weight") -> None:
    if weight.size(-1) % 4 != 0:
        raise ValueError(f"{name} last dimension must be divisible by 4 for 2:4 structured sparsity")
    if not is_2to4_sparse(weight):
        raise ValueError(f"{name} must satisfy 2:4 structured sparsity")


def magnitude_prune_2to4_bf16(weight: torch.Tensor) -> torch.Tensor:
    if weight.size(-1) % 4 != 0:
        raise ValueError("weight last dimension must be divisible by 4 for 2:4 pruning")
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    groups = weight_bf16.reshape(-1, 4)
    keep = torch.zeros_like(groups, dtype=torch.bool)
    top2 = groups.abs().topk(k=2, dim=1).indices
    keep.scatter_(1, top2, True)
    return torch.where(keep, groups, torch.zeros_like(groups)).reshape_as(weight_bf16)


def cusparselt_version() -> int:
    return int(_load_ext().cusparselt_version())


def pack_sparse_bf16_a_from_dense(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _check_2d_cuda_bf16(a, "a")
    require_2to4_sparse(a, "a")
    return _load_ext().sparse_bf16_compress_weight(a)


def pack_spinfer_sparse_bf16_a_from_dense(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    _check_2d_cuda_bf16(a, "a")
    require_2to4_sparse(a, "a")
    _check_spinfer_sparse_bf16_supported_shape(int(a.size(0)), 8, int(a.size(1)))
    compressed, tile_offsets_global, tile_offsets_median, bitmap, max_nnz_intile = (
        _load_ext().spinfer_sparse_bf16_compress_weight(a)
    )
    return compressed, tile_offsets_global, tile_offsets_median, bitmap, int(max_nnz_intile)


def sparse_bf16_gemm_bf16(
    m: int,
    sparse_a: torch.Tensor,
    metadata: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor | float = 1.0,
    *,
    backend: str = "cusparselt",
    tile_offsets_median: torch.Tensor | None = None,
    bitmap: torch.Tensor | None = None,
    max_nnz_intile: int = 0,
) -> torch.Tensor:
    _check_2d_cuda_bf16(b, "b")
    backend = _normalize_sparse_bf16_backend(backend)
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            raise ValueError("sparse BF16 alpha must contain exactly one element")
        alpha_value = float(alpha.detach().cpu().reshape(1).item())
    else:
        alpha_value = float(alpha)
    if alpha_value != 1.0:
        raise ValueError("sparse BF16 wrapper only supports alpha=1.0")
    n = int(b.size(0))
    k = int(b.size(1))
    if backend == "cusparselt":
        del metadata, tile_offsets_median, bitmap, max_nnz_intile
        _check_sparse_bf16_supported_shape(int(m), n, k)
        return _load_ext().sparse_bf16_matmul(
            int(m),
            int(n),
            int(k),
            sparse_a.contiguous(),
            b.contiguous(),
        )

    if tile_offsets_median is None or bitmap is None:
        raise ValueError("SpInfer sparse BF16 requires tile_offsets_median and bitmap")
    _check_spinfer_sparse_bf16_supported_shape(int(m), n, k)
    return _load_ext().spinfer_sparse_bf16_matmul(
        int(m),
        int(n),
        int(k),
        sparse_a.contiguous(),
        metadata.contiguous(),
        tile_offsets_median.contiguous(),
        bitmap.contiguous(),
        int(max_nnz_intile),
        b.contiguous(),
    )


@dataclass(frozen=True)
class SparseBF16Weight:
    sparse_weight: torch.Tensor
    metadata: torch.Tensor
    in_features: int
    out_features: int
    original_dtype: torch.dtype
    bias: torch.Tensor | None = None
    backend: str = "cusparselt"
    tile_offsets_median: torch.Tensor | None = None
    bitmap: torch.Tensor | None = None
    max_nnz_intile: int = 0


@torch.no_grad()
def pack_sparse_bf16_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    original_dtype: torch.dtype | None = None,
    prune: bool = False,
    backend: str = "cusparselt",
) -> SparseBF16Weight:
    _check_2d_cuda_bf16(weight, "weight")
    backend = _normalize_sparse_bf16_backend(backend)
    out_features, in_features = int(weight.size(0)), int(weight.size(1))
    if backend == "cusparselt":
        _check_sparse_bf16_supported_shape(out_features, 8, in_features)
    else:
        _check_spinfer_sparse_bf16_supported_shape(out_features, 8, in_features)

    sparse_weight_bf16 = weight.contiguous()
    if prune:
        sparse_weight_bf16 = magnitude_prune_2to4_bf16(sparse_weight_bf16).contiguous()
    else:
        require_2to4_sparse(sparse_weight_bf16, "weight")

    tile_offsets_median = None
    bitmap = None
    max_nnz_intile = 0
    if backend == "cusparselt":
        sparse_weight, metadata = pack_sparse_bf16_a_from_dense(sparse_weight_bf16)
    else:
        sparse_weight, metadata, tile_offsets_median, bitmap, max_nnz_intile = pack_spinfer_sparse_bf16_a_from_dense(
            sparse_weight_bf16
        )
    packed_bias = None
    if bias is not None:
        if bias.dim() != 1 or bias.numel() != out_features:
            raise ValueError("bias must have shape [out_features]")
        if bias.device != weight.device:
            raise ValueError("bias must be on the same device as weight")
        packed_bias = bias.detach().to(device=weight.device, dtype=torch.bfloat16).contiguous()

    return SparseBF16Weight(
        sparse_weight=sparse_weight,
        metadata=metadata,
        in_features=in_features,
        out_features=out_features,
        original_dtype=original_dtype or weight.dtype,
        bias=packed_bias,
        backend=backend,
        tile_offsets_median=tile_offsets_median,
        bitmap=bitmap,
        max_nnz_intile=max_nnz_intile,
    )


class SparseBF16Linear(nn.Module):
    """Inference-only structured sparse BF16 replacement for torch.nn.Linear."""

    in_features: int
    out_features: int

    def __init__(self, weight: SparseBF16Weight) -> None:
        super().__init__()
        self.in_features = weight.in_features
        self.out_features = weight.out_features
        self.original_dtype = weight.original_dtype
        self.backend = weight.backend
        self.max_nnz_intile = int(weight.max_nnz_intile)
        self.register_buffer("sparse_weight", weight.sparse_weight.contiguous(), persistent=True)
        self.register_buffer("metadata", weight.metadata.contiguous(), persistent=True)
        if weight.tile_offsets_median is None:
            self.register_buffer("tile_offsets_median", None, persistent=True)
        else:
            self.register_buffer("tile_offsets_median", weight.tile_offsets_median.contiguous(), persistent=True)
        if weight.bitmap is None:
            self.register_buffer("bitmap", None, persistent=True)
        else:
            self.register_buffer("bitmap", weight.bitmap.contiguous(), persistent=True)
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
        prune: bool = False,
        backend: str = "cusparselt",
    ) -> "SparseBF16Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an instance of torch.nn.Linear")
        target_device = torch.device(device) if device is not None else linear.weight.device
        if target_device.type != "cuda":
            raise ValueError("SparseBF16Linear.from_linear requires a CUDA target device")
        weight = linear.weight.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(device=target_device, dtype=torch.bfloat16).contiguous()
        return cls(pack_sparse_bf16_weight(weight, bias, original_dtype=linear.weight.dtype, prune=prune, backend=backend))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("SparseBF16Linear is inference-only; call eval() before forward")
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("SparseBF16Linear does not implement autograd; run under torch.no_grad()")
        if x.device.type != "cuda":
            raise ValueError("x must be a CUDA tensor")
        if x.device != self.sparse_weight.device:
            raise ValueError("x must be on the same CUDA device as the SparseBF16 weight")
        if x.size(-1) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {x.size(-1)}")

        batch_shape = x.shape[:-1]
        with torch.no_grad():
            x_flat = x.to(dtype=torch.bfloat16).reshape(-1, self.in_features).contiguous()
            tokens = int(x_flat.size(0))
            if self.backend == "cusparselt":
                _check_sparse_bf16_supported_shape(self.out_features, tokens, self.in_features)
                can_use_backend = can_use_cusparselt_sparse_bf16(self.out_features, tokens, self.in_features)
                backend_name = "cuSPARSELt"
            else:
                _check_spinfer_sparse_bf16_supported_shape(self.out_features, tokens, self.in_features)
                can_use_backend = can_use_spinfer_sparse_bf16(self.out_features, tokens, self.in_features)
                backend_name = "SpInfer"
            if not can_use_backend:
                raise RuntimeError(
                    f"{backend_name} sparse BF16 GEMM cannot implement M={self.out_features}, "
                    f"N={tokens}, K={self.in_features}"
                )

            out = sparse_bf16_gemm_bf16(
                self.out_features,
                self.sparse_weight,
                self.metadata,
                x_flat,
                1.0,
                backend=self.backend,
                tile_offsets_median=self.tile_offsets_median,
                bitmap=self.bitmap,
                max_nnz_intile=self.max_nnz_intile,
            )
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*batch_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self.backend}"
        )


class PaddedSparseBF16Linear(nn.Module):
    """Sparse BF16 Linear wrapper that pads the flattened token dimension for cuSPARSELt."""

    in_features: int
    out_features: int

    def __init__(self, sparse_linear: SparseBF16Linear, pad_multiple: int = 8) -> None:
        super().__init__()
        if not isinstance(sparse_linear, SparseBF16Linear):
            raise TypeError("sparse_linear must be a SparseBF16Linear")
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be positive")
        self.sparse_linear = sparse_linear
        self.pad_multiple = int(pad_multiple)
        self.in_features = sparse_linear.in_features
        self.out_features = sparse_linear.out_features
        self.original_dtype = sparse_linear.original_dtype
        self.backend = sparse_linear.backend
        self.eval()
        self.requires_grad_(False)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        device: torch.device | str | None = None,
        prune: bool = False,
        pad_multiple: int = 8,
        backend: str = "cusparselt",
    ) -> "PaddedSparseBF16Linear":
        return cls(
            SparseBF16Linear.from_linear(linear, device=device, prune=prune, backend=backend),
            pad_multiple=pad_multiple,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(f"expected input last dim {self.in_features}, got {x.size(-1)}")
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
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
            f"pad_multiple={self.pad_multiple}, backend={self.backend}"
        )


def simulate_sparse_bf16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    prune: bool = False,
) -> torch.Tensor:
    if x.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("simulate_sparse_bf16_linear expects CUDA tensors")
    weight_bf16 = weight.to(torch.bfloat16).contiguous()
    if prune:
        weight_bf16 = magnitude_prune_2to4_bf16(weight_bf16).contiguous()
    else:
        require_2to4_sparse(weight_bf16, "weight")
    bias_bf16 = bias.to(torch.bfloat16).contiguous() if bias is not None else None
    return F.linear(x.to(torch.bfloat16), weight_bf16, bias_bf16)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


__all__ = [
    "PaddedSparseBF16Linear",
    "SparseBF16Linear",
    "SparseBF16Weight",
    "can_use_cusparselt_sparse_bf16",
    "can_use_cutlass_sparse_bf16",
    "can_use_spinfer_sparse_bf16",
    "cusparselt_paths",
    "cusparselt_version",
    "is_2to4_sparse",
    "magnitude_prune_2to4_bf16",
    "pack_sparse_bf16_a_from_dense",
    "pack_spinfer_sparse_bf16_a_from_dense",
    "pack_sparse_bf16_weight",
    "require_2to4_sparse",
    "simulate_sparse_bf16_linear",
    "sparse_bf16_gemm_bf16",
]
