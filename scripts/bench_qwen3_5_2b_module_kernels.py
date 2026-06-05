#!/usr/bin/env python
"""Benchmark packaged kernel Linear.forward latency for Qwen3.5 linear shapes.

This is model-shape specific. It tests the unique compressible Linear groups in
one Qwen3.5 model while sweeping token M.  The measured callable is module(x), not raw
GEMM, so activation packing and wrapper overhead are included.

CSV semantics:
  M = flattened token count
  N = out_features
  K = in_features

Sparse modules use the padded wrappers used by Qwen runtime, so small M values
are padded before calling the underlying sparse kernels.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.cutlass_sparse_bf16 import (  # noqa: E402
    PaddedSparseBF16Linear,
    SPARSE_BF16_BLOCKED_SHAPES,
)
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear  # noqa: E402


KERNELS = ["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"]
DEFAULT_M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_MODEL_NAME = "Qwen3.5-2B"
DEFAULT_MODEL_ROOT = Path("/home/agent/wja/data/models/Qwen")
DEFAULT_OUTPUT_ROOT = Path("artifacts/results/benchmarks/module")
LINEAR_SUFFIXES = [
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]


@dataclass(frozen=True)
class LinearShape:
    group: str
    count: int
    n: int
    k: int


def _model_slug(model_name: str) -> str:
    return model_name.lower().replace(".", "").replace("-", "_")


def _default_output(model_name: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / model_name / "kernel" / f"{_model_slug(model_name)}_module_kernel_curves.csv"


def _default_model_path(model_name: str) -> Path:
    return DEFAULT_MODEL_ROOT / model_name


def _extract_linear_shapes(model_path: Path) -> list[LinearShape]:
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensors files found under {model_path}")

    pattern = re.compile(r"^model\.language_model\.layers\.\d+\.(.+)\.weight$")
    counts: dict[str, dict[tuple[int, int], int]] = {suffix: {} for suffix in LINEAR_SUFFIXES}
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                match = pattern.match(name)
                if not match:
                    continue
                suffix = match.group(1)
                if suffix not in counts:
                    continue
                shape = tuple(int(v) for v in tensors.get_slice(name).get_shape())
                if len(shape) != 2:
                    continue
                counts[suffix][shape] = counts[suffix].get(shape, 0) + 1

    shapes: list[LinearShape] = []
    for suffix in LINEAR_SUFFIXES:
        group_shapes = counts[suffix]
        if not group_shapes:
            continue
        if len(group_shapes) != 1:
            raise RuntimeError(f"Multiple shapes for {suffix}: {group_shapes}")
        (n, k), count = next(iter(group_shapes.items()))
        shapes.append(LinearShape(suffix, count, n, k))
    if not shapes:
        raise RuntimeError(f"No compressible Qwen3.5 linear shapes found under {model_path}")
    return shapes


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


def _make_base_linear(n: int, k: int, device: torch.device, *, bias: bool, seed: int) -> nn.Linear:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    linear = nn.Linear(k, n, bias=bias, device=device, dtype=torch.bfloat16)
    linear.weight.data.normal_(mean=0.0, std=0.02, generator=generator)
    if bias:
        linear.bias.data.normal_(mean=0.0, std=0.02, generator=generator)
    linear.eval()
    linear.requires_grad_(False)
    return linear


def _kernel_supported(kernel: str, n: int, k: int) -> tuple[bool, str]:
    wrapper = _load_wrapper()
    if kernel == "dense_bf16":
        return True, ""
    if kernel == "dense_nvfp4":
        if not wrapper.can_use_cutlass_nvfp4(1, n, k, load_extension=False):
            return False, f"shape_not_supported:dense_nvfp4:N={n},K={k}"
        return True, ""
    if kernel == "marlin_nvfp4":
        if not wrapper.can_use_marlin_nvfp4(1, n, k, load_extension=False):
            return False, f"shape_not_supported:marlin_nvfp4:N={n},K={k}"
        return True, ""
    if kernel == "sparse_bf16":
        if (n, k) in SPARSE_BF16_BLOCKED_SHAPES:
            return False, f"shape_blocked:sparse_bf16:N={n},K={k}"
        if not wrapper.can_use_cutlass_sparse_bf16(n, 8, k, load_extension=False):
            return False, f"shape_not_supported:sparse_bf16:N={n},K={k}"
        return True, ""
    if kernel == "sparse_nvfp4":
        if not wrapper.can_use_cutlass_sparse_nvfp4(n, 32, k, load_extension=False):
            return False, f"shape_not_supported:sparse_nvfp4:N={n},K={k}"
        return True, ""
    raise ValueError(f"unknown kernel: {kernel}")


@torch.no_grad()
def _make_module(kernel: str, base_linear: nn.Linear, device: torch.device, dtype: torch.dtype) -> nn.Module:
    wrapper = _load_wrapper()
    if kernel == "dense_bf16":
        return base_linear
    if kernel == "dense_nvfp4":
        return wrapper.NVFP4Linear.from_linear(base_linear, device=device).eval()
    if kernel == "marlin_nvfp4":
        return wrapper.MarlinNVFP4Linear.from_linear(base_linear, device=device, activation_dtype=dtype).eval()
    if kernel == "sparse_bf16":
        sparse = wrapper.SparseBF16Linear.from_linear(base_linear, device=device, prune=True).eval()
        return PaddedSparseBF16Linear(sparse, 8).eval()
    if kernel == "sparse_nvfp4":
        sparse = wrapper.SparseNVFP4Linear.from_linear(base_linear, device=device, prune=True).eval()
        return PaddedSparseNVFP4Linear(sparse, 32).eval()
    raise ValueError(f"unknown kernel: {kernel}")


def _time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[float, float, float, torch.Tensor]:
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()

    times: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_event.record()
        result = fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(float(start_event.elapsed_time(end_event)))
    if result is None:
        raise RuntimeError("timed callable produced no result")
    mean_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    return mean_ms, min_ms, max_ms, result


def _tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return (2.0 * m * n * k) / (latency_ms * 1e-3) / 1e12


def _classify_exception(exc: Exception) -> tuple[str, str]:
    msg = " ".join(str(exc).split())
    lower = msg.lower()
    if "out of memory" in lower or "oom" in lower:
        return "skip", "CUDA OOM"
    if "cusparse status 10" in lower:
        return "skip", "cuSPARSELt unsupported shape"
    if "sparse_nvf4_pack_b failed" in msg:
        return "skip", "sparse_nvfp4 pack_b failed"
    if "qutlass_sparse_nvf4_gemm_from_sparse failed with status -1" in msg:
        return "skip", "sparse_nvfp4 unsupported shape"
    return "error", msg[:500]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "benchmark_level",
        "linear_group",
        "linear_count",
        "m",
        "n",
        "k",
        "kernel",
        "status",
        "latency_ms",
        "latency_min_ms",
        "latency_max_ms",
        "tflops",
        "speedup_vs_bf16",
        "padded_m",
        "gpu",
        "warmup",
        "iters",
        "bias",
        "torch_version",
        "cuda_version",
        "error_msg",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _padded_m(kernel: str, m: int) -> int:
    if kernel == "sparse_bf16":
        return ((m + 7) // 8) * 8
    if kernel == "sparse_nvfp4":
        return ((m + 31) // 32) * 32
    return m


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path) if args.model_path else _default_model_path(args.model_name)
    if args.print_shapes:
        linear_shapes = _extract_linear_shapes(model_path)
        print(f"Model: {args.model_name}")
        print(f"Model path: {model_path}")
        print("linear_group,linear_count,n,k")
        for shape in linear_shapes:
            print(f"{shape.group},{shape.count},{shape.n},{shape.k}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {args.gpu} not available; found {torch.cuda.device_count()}")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    output = Path(args.output) if args.output else _default_output(args.model_name)
    linear_shapes = _extract_linear_shapes(model_path)
    rows: list[dict[str, object]] = []

    print(f"Output: {output}")
    print(f"Model: {args.model_name}")
    print(f"Model path: {model_path}")
    print(f"GPU: {args.gpu} {torch.cuda.get_device_name(device)}")
    print(f"M values: {args.m_values}")

    for shape_idx, shape in enumerate(linear_shapes, 1):
        print(f"[{shape_idx}/{len(linear_shapes)}] {shape.group} count={shape.count} N={shape.n} K={shape.k}")
        base_linear = _make_base_linear(
            shape.n,
            shape.k,
            device,
            bias=args.bias,
            seed=args.seed + shape_idx,
        )
        modules: dict[str, nn.Module] = {}
        module_errors: dict[str, tuple[str, str]] = {}
        for kernel in KERNELS:
            supported, reason = _kernel_supported(kernel, shape.n, shape.k)
            if not supported:
                module_errors[kernel] = ("skip", reason)
                continue
            try:
                modules[kernel] = _make_module(kernel, base_linear, device, dtype)
            except Exception as exc:
                module_errors[kernel] = _classify_exception(exc)

        for m in args.m_values:
            x = torch.randn((1, m, shape.k), device=device, dtype=dtype)
            bf16_latency = None
            m_rows: list[dict[str, object]] = []

            for kernel in KERNELS:
                base = {
                    "model": args.model_name,
                    "benchmark_level": "module_forward",
                    "linear_group": shape.group,
                    "linear_count": shape.count,
                    "m": m,
                    "n": shape.n,
                    "k": shape.k,
                    "kernel": kernel,
                    "padded_m": _padded_m(kernel, m),
                    "gpu": args.gpu,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "bias": args.bias,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                }

                if kernel in module_errors:
                    status, err = module_errors[kernel]
                    m_rows.append({**base, "status": status, "error_msg": err})
                    continue

                try:
                    latency_ms, min_ms, max_ms, result = _time_cuda(lambda: modules[kernel](x), args.warmup, args.iters)
                    if not torch.isfinite(result.float()).all().item():
                        raise RuntimeError("module output contains NaN/Inf")
                    row = {
                        **base,
                        "status": "pass",
                        "latency_ms": f"{latency_ms:.6f}",
                        "latency_min_ms": f"{min_ms:.6f}",
                        "latency_max_ms": f"{max_ms:.6f}",
                        "tflops": f"{_tflops(m, shape.n, shape.k, latency_ms):.6f}",
                        "error_msg": "",
                    }
                    if kernel == "dense_bf16":
                        bf16_latency = latency_ms
                    m_rows.append(row)
                except Exception as exc:
                    status, err = _classify_exception(exc)
                    m_rows.append({**base, "status": status, "error_msg": err})
                    if status == "error":
                        print(f"  ERROR M={m} kernel={kernel}: {err[:120]}")
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

            if bf16_latency and bf16_latency > 0:
                for row in m_rows:
                    if row.get("status") == "pass" and row.get("latency_ms"):
                        speedup = bf16_latency / float(row["latency_ms"])
                        row["speedup_vs_bf16"] = f"{speedup:.6f}"
            rows.extend(m_rows)
            print(f"  M={m} done")

        del modules, base_linear
        gc.collect()
        torch.cuda.empty_cache()

    _write_rows(output, rows)
    print(f"Wrote {output} rows={len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--print-shapes", action="store_true", help="Print model linear shapes and exit without CUDA.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--m-values", nargs="+", type=int, default=DEFAULT_M_VALUES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bias", action="store_true", help="Use Linear bias. Default is no bias, matching Qwen linears.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
