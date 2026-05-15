#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fake.kernels.flashinfer_nvfp4 import _global_scale as nvfp4_global_scale
from fake.kernels.flashinfer_nvfp4 import _load_flashinfer, flashinfer_version
from fake.utils.csv_io import append_csv_row


DEFAULT_OUTPUT = "artifacts/analysis/flashinfer/custom_shapes.csv"
M_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
NK_SWEEP = [512, 1024, 2048, 4096, 8192, 16384]
SQUARE_SWEEP = [512, 1024, 2048, 4096, 8192, 16384]


@dataclass(frozen=True)
class BenchShape:
    family: str
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class BenchStats:
    mean_ms: float
    min_ms: float
    max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FlashInfer NVFP4 custom GEMM shapes.")
    parser.add_argument("--preset", choices=["smoke", "balanced", "large"], default="balanced")
    parser.add_argument("--shapes", nargs="*", default=None, help="Optional custom shapes as MxNxK.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--gemm-backend", choices=["auto", "cutlass", "cudnn", "trtllm", "cute-dsl", "b12x"], default="auto")
    parser.add_argument("--quant-backend", choices=["cuda", "cute-dsl"], default="cuda")
    parser.add_argument("--sf-layout", choices=["layout_128x4", "layout_8x4", "layout_linear"], default="layout_128x4")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >= 0 and --iters must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    flashinfer = _load_flashinfer()
    sf_layout = _resolve_sf_layout(flashinfer, args.sf_layout)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shapes = _parse_custom_shapes(args.shapes) if args.shapes else _preset_shapes(args.preset)

    print(
        "flashinfer custom shape bench: "
        f"preset={args.preset} shapes={len(shapes)} dtype={args.dtype} "
        f"gemm_backend={args.gemm_backend} quant_backend={args.quant_backend} output={args.output}"
    )
    for shape_index, shape in enumerate(shapes):
        base = _base_row(args, shape_index, shape, device)
        try:
            _validate_shape(shape)
            _benchmark_shape(args, flashinfer, sf_layout, dtype, device, shape, base)
            print(f"OK {shape.family} m={shape.m} n={shape.n} k={shape.k}")
        except Exception as exc:
            _write_error_row(args.output, base, exc)
            _cleanup_cuda()
            print(f"ERROR {shape.family} m={shape.m} n={shape.n} k={shape.k}: {type(exc).__name__}: {exc}")


def _benchmark_shape(
    args: argparse.Namespace,
    flashinfer,
    sf_layout,
    dtype: torch.dtype,
    device: torch.device,
    shape: BenchShape,
    base: dict[str, object],
) -> None:
    x = torch.randn((shape.m, shape.k), device=device, dtype=dtype)
    weight = torch.randn((shape.n, shape.k), device=device, dtype=dtype)
    x_fp32 = x.float()
    weight_fp32 = weight.float()

    weight_shuffle = args.gemm_backend == "trtllm"
    weight_global_sf, weight_fp4, weight_sf = _quantize_nvfp4(
        flashinfer, weight, sf_layout, weight_shuffle, args.quant_backend
    )
    weight_fp4_t = weight_fp4.t()
    weight_sf_t = weight_sf.t()

    activation_global_sf = nvfp4_global_scale(x)
    activation_fp4, activation_sf = _activation_quant(flashinfer, x, activation_global_sf, sf_layout, args.quant_backend)
    alpha = _alpha(activation_global_sf, weight_global_sf)

    op_stats: dict[str, BenchStats] = {
        "dense_linear_bf16": _bench_cuda(lambda: F.linear(x, weight), args.warmup, args.iters),
        "dense_linear_fp32": _bench_cuda(lambda: F.linear(x_fp32, weight_fp32), args.warmup, args.iters),
        "activation_global_scale": _bench_cuda(lambda: nvfp4_global_scale(x), args.warmup, args.iters),
        "activation_quant_only": _bench_cuda(
            lambda: _activation_quant(flashinfer, x, activation_global_sf, sf_layout, args.quant_backend),
            args.warmup,
            args.iters,
        ),
        "activation_scale_plus_quant": _bench_cuda(
            lambda: _quantize_nvfp4(flashinfer, x, sf_layout, False, args.quant_backend),
            args.warmup,
            args.iters,
        ),
        "weight_scale_plus_quant_once": _bench_cuda(
            lambda: _quantize_nvfp4(flashinfer, weight, sf_layout, weight_shuffle, args.quant_backend),
            args.warmup,
            args.iters,
        ),
        "alpha": _bench_cuda(lambda: _alpha(activation_global_sf, weight_global_sf), args.warmup, args.iters),
        "nvfp4_gemm_only": _bench_cuda(
            lambda: _mm_fp4(
                flashinfer,
                activation_fp4,
                weight_fp4_t,
                activation_sf,
                weight_sf_t,
                alpha,
                dtype,
                args.sf_layout,
                args.gemm_backend,
            ),
            args.warmup,
            args.iters,
        ),
        "nvfp4_forward_like": _bench_cuda(
            lambda: _forward_like(
                flashinfer,
                x,
                weight_global_sf,
                weight_fp4_t,
                weight_sf_t,
                sf_layout,
                dtype,
                args.sf_layout,
                args.gemm_backend,
                args.quant_backend,
            ),
            args.warmup,
            args.iters,
        ),
    }
    dense_bf16_ms = op_stats["dense_linear_bf16"].mean_ms
    dense_fp32_ms = op_stats["dense_linear_fp32"].mean_ms
    nvfp4_forward_ms = op_stats["nvfp4_forward_like"].mean_ms
    nvfp4_gemm_ms = op_stats["nvfp4_gemm_only"].mean_ms

    for op, stats in op_stats.items():
        _write_row(
            args.output,
            base,
            {
                "status": "OK",
                "error_type": "",
                "error_message": "",
                "op": op,
                **_stats_fields(stats),
                "speedup_vs_dense_bf16": _ratio(dense_bf16_ms, stats.mean_ms),
                "speedup_vs_dense_fp32": _ratio(dense_fp32_ms, stats.mean_ms),
                "nvfp4_forward_speedup_vs_dense_bf16": _ratio(dense_bf16_ms, nvfp4_forward_ms),
                "nvfp4_forward_speedup_vs_dense_fp32": _ratio(dense_fp32_ms, nvfp4_forward_ms),
                "nvfp4_gemm_speedup_vs_dense_bf16": _ratio(dense_bf16_ms, nvfp4_gemm_ms),
                "nvfp4_gemm_speedup_vs_dense_fp32": _ratio(dense_fp32_ms, nvfp4_gemm_ms),
            },
        )


def _forward_like(
    flashinfer,
    x: torch.Tensor,
    weight_global_sf: torch.Tensor,
    weight_fp4_t: torch.Tensor,
    weight_sf_t: torch.Tensor,
    sf_layout,
    out_dtype: torch.dtype,
    sf_layout_name: str,
    gemm_backend: str,
    quant_backend: str,
) -> torch.Tensor:
    activation_global_sf, activation_fp4, activation_sf = _quantize_nvfp4(
        flashinfer, x, sf_layout, False, quant_backend
    )
    alpha = _alpha(activation_global_sf, weight_global_sf)
    return _mm_fp4(
        flashinfer,
        activation_fp4,
        weight_fp4_t,
        activation_sf,
        weight_sf_t,
        alpha,
        out_dtype,
        sf_layout_name,
        gemm_backend,
    )


def _quantize_nvfp4(flashinfer, x: torch.Tensor, sf_layout, do_shuffle: bool, backend: str):
    global_sf = nvfp4_global_scale(x)
    x_fp4, x_sf = _activation_quant(flashinfer, x, global_sf, sf_layout, backend, do_shuffle=do_shuffle)
    return global_sf, x_fp4, x_sf


def _activation_quant(
    flashinfer,
    x: torch.Tensor,
    global_sf: torch.Tensor,
    sf_layout,
    backend: str,
    do_shuffle: bool = False,
):
    return flashinfer.nvfp4_quantize(
        x,
        global_sf,
        sfLayout=sf_layout,
        do_shuffle=do_shuffle,
        sf_vec_size=16,
        backend=backend,
    )


def _alpha(activation_global_sf: torch.Tensor, weight_global_sf: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(activation_global_sf * weight_global_sf)


def _mm_fp4(
    flashinfer,
    activation_fp4: torch.Tensor,
    weight_fp4_t: torch.Tensor,
    activation_sf: torch.Tensor,
    weight_sf_t: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    sf_layout_name: str,
    backend: str,
) -> torch.Tensor:
    return flashinfer.mm_fp4(
        activation_fp4,
        weight_fp4_t,
        activation_sf,
        weight_sf_t,
        alpha,
        out_dtype,
        None,
        16,
        sf_layout_name == "layout_8x4",
        backend,
        True,
    )


def _bench_cuda(fn: Callable[[], object], warmup: int, iters: int) -> BenchStats:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    return BenchStats(mean_ms=sum(times) / len(times), min_ms=min(times), max_ms=max(times))


def _preset_shapes(preset: str) -> list[BenchShape]:
    if preset == "smoke":
        return [
            BenchShape("smoke_square", 512, 512, 512),
            BenchShape("smoke_dinov3_like", 256, 4096, 4096),
            BenchShape("smoke_maxvit_like", 4096, 2048, 512),
        ]

    shapes = _balanced_shapes()
    if preset == "balanced":
        return shapes

    shapes.extend(
        [
            BenchShape("m_sweep_large_context", 131072, 4096, 4096),
            BenchShape("n_sweep_large_context", 4096, 32768, 4096),
            BenchShape("k_sweep_large_context", 4096, 4096, 32768),
            BenchShape("square_scale_sweep", 32768, 32768, 32768),
        ]
    )
    return shapes


def _balanced_shapes() -> list[BenchShape]:
    shapes: list[BenchShape] = []
    for m in M_SWEEP:
        shapes.append(BenchShape("m_sweep_large_context", m, 4096, 4096))
        shapes.append(BenchShape("m_sweep_small_context", m, 512, 512))
    for n in NK_SWEEP:
        shapes.append(BenchShape("n_sweep_large_context", 4096, n, 4096))
        shapes.append(BenchShape("n_sweep_small_context", 512, n, 512))
    for k in NK_SWEEP:
        shapes.append(BenchShape("k_sweep_large_context", 4096, 4096, k))
        shapes.append(BenchShape("k_sweep_small_context", 512, 512, k))
    for n in NK_SWEEP:
        shapes.append(BenchShape("compute_reuse_sweep", 4096, n, 4096))
    for size in SQUARE_SWEEP:
        shapes.append(BenchShape("square_scale_sweep", size, size, size))
    shapes.extend(
        [
            BenchShape("model_anchor_dinov3", 69, 4096, 4096),
            BenchShape("model_anchor_dinov3", 2088, 8192, 4096),
            BenchShape("model_anchor_dinov3", 33408, 8192, 4096),
            BenchShape("model_anchor_dinov3", 74368, 8192, 4096),
            BenchShape("model_anchor_dinov3", 74368, 4096, 8192),
            BenchShape("model_anchor_maxvit", 3136, 64, 64),
            BenchShape("model_anchor_maxvit", 28224, 2048, 512),
            BenchShape("model_anchor_maxvit", 28224, 3072, 768),
            BenchShape("model_anchor_maxvit", 25088, 3072, 768),
        ]
    )
    return shapes


def _parse_custom_shapes(values: list[str]) -> list[BenchShape]:
    shapes = []
    for value in values:
        m, n, k = _parse_shape(value)
        shapes.append(BenchShape("custom", m, n, k))
    return shapes


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected MxNxK shape, got: {value}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _validate_shape(shape: BenchShape) -> None:
    if shape.m <= 0 or shape.n <= 0 or shape.k <= 0:
        raise ValueError(f"Shape dimensions must be positive, got m={shape.m} n={shape.n} k={shape.k}")
    if shape.k % 16 != 0:
        raise ValueError(f"FlashInfer NVFP4 requires k to be divisible by 16, got k={shape.k}")


def _base_row(args: argparse.Namespace, shape_index: int, shape: BenchShape, device: torch.device) -> dict[str, object]:
    flops = 2 * shape.m * shape.n * shape.k
    bytes_bf16 = 2 * (shape.m * shape.k + shape.n * shape.k + shape.m * shape.n)
    bytes_fp32 = 4 * (shape.m * shape.k + shape.n * shape.k + shape.m * shape.n)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": "flashinfer_nvfp4_custom_shape",
        "task": "custom_shape_microbench",
        "preset": args.preset,
        "shape_index": shape_index,
        "shape_family": shape.family,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "flops": flops,
        "estimated_dense_bf16_bytes": bytes_bf16,
        "estimated_dense_fp32_bytes": bytes_fp32,
        "arithmetic_intensity_bf16": _format_float(flops / bytes_bf16),
        "arithmetic_intensity_fp32": _format_float(flops / bytes_fp32),
        "dtype_arg": args.dtype,
        "runtime_dtype": "bfloat16" if args.dtype == "bf16" else "float16",
        "device": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_module": os.environ.get("CUDA_MODULE", ""),
        "flashinfer_version": flashinfer_version(),
        "nvfp4_block_size": 16,
        "nvfp4_gemm_backend": args.gemm_backend,
        "nvfp4_quant_backend": args.quant_backend,
        "nvfp4_sf_layout": args.sf_layout,
    }


def _resolve_sf_layout(flashinfer, sf_layout: str):
    try:
        return getattr(flashinfer.SfLayout, sf_layout)
    except AttributeError as exc:
        choices = [name for name in dir(flashinfer.SfLayout) if name.startswith("layout_")]
        raise ValueError(f"Unsupported FlashInfer scale layout: {sf_layout}. Choices: {choices}") from exc


def _write_row(output: str, base: dict[str, object], fields: dict[str, object]) -> None:
    row = {**base, **fields}
    append_csv_row(output, list(row.keys()), row)


def _write_error_row(output: str, base: dict[str, object], exc: Exception) -> None:
    _write_row(
        output,
        base,
        {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error_message": _short_error_message(exc),
            "op": "error",
            "latency_mean_ms": "",
            "latency_min_ms": "",
            "latency_max_ms": "",
            "speedup_vs_dense_bf16": "",
            "speedup_vs_dense_fp32": "",
            "nvfp4_forward_speedup_vs_dense_bf16": "",
            "nvfp4_forward_speedup_vs_dense_fp32": "",
            "nvfp4_gemm_speedup_vs_dense_bf16": "",
            "nvfp4_gemm_speedup_vs_dense_fp32": "",
        },
    )


def _stats_fields(stats: BenchStats) -> dict[str, str]:
    return {
        "latency_mean_ms": f"{stats.mean_ms:.6f}",
        "latency_min_ms": f"{stats.min_ms:.6f}",
        "latency_max_ms": f"{stats.max_ms:.6f}",
    }


def _ratio(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return ""
    return _format_float(numerator / denominator)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _short_error_message(exc: Exception, limit: int = 500) -> str:
    message = " ".join(str(exc).split())
    return message[:limit]


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
