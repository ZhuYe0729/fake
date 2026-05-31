#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fake.compression.pruning import PruneResult, prune_dense_2_4, prune_nvfp4_pair_2_4
from fake.kernels.cutlass_nvfp4 import _load_cutlass_nvfp4_symbols
from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear, _load_cutlass_sparse_bf16_symbols
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear, _load_cutlass_sparse_nvfp4_symbols
from fake.utils.csv_io import append_csv_row


DEFAULT_OUTPUT_DIR = "artifacts/analysis/linear_kernel_shape_sweep"
DEFAULT_FIXED_VALUES = "1,16,64,256,4096,16384"
DEFAULT_VARIABLE_VALUES = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384"
METHODS = ("dense_fp32", "dense_bf16", "sparse_bf16", "dense_nvfp4", "sparse_nvfp4")
UNSUPPORTED_EXCEPTIONS = (ValueError, TypeError)


@dataclass(frozen=True)
class BenchStats:
    mean_ms: float
    p50_ms: float
    min_ms: float
    max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real Linear kernels over an M/N/K shape sweep.")
    parser.add_argument("--fixed-dim", choices=["m", "n", "k"], required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fixed-values", default=DEFAULT_FIXED_VALUES)
    parser.add_argument("--variable-values", default=DEFAULT_VARIABLE_VALUES)
    parser.add_argument("--resume", action="store_true", help="Skip shape/method rows already present in --output.")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >= 0 and --iters must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    fixed_values = _parse_int_list(args.fixed_values, "--fixed-values")
    variable_values = _parse_int_list(args.variable_values, "--variable-values")
    output = args.output or str(Path(DEFAULT_OUTPUT_DIR) / f"fixed_{args.fixed_dim}.csv")
    existing = _load_existing_rows(output) if args.resume else {}

    print(
        "linear kernel shape sweep: "
        f"fixed_dim={args.fixed_dim} fixed_values={fixed_values} "
        f"variable_values={variable_values} warmup={args.warmup} iters={args.iters} output={output}"
    )

    for shape_index, (m, n, k) in enumerate(_iter_shapes(args.fixed_dim, fixed_values, variable_values)):
        base = _base_row(args, output, shape_index, m, n, k, device)
        existing_shape = existing.get(shape_index, {})
        methods_to_run = [method for method in METHODS if method not in existing_shape]
        if not methods_to_run:
            print(f"SKIP shape_index={shape_index} m={m} n={n} k={k}")
            continue
        try:
            _benchmark_shape(args, output, base, m, n, k, device, methods_to_run, existing_shape)
            print(f"OK shape_index={shape_index} m={m} n={n} k={k} methods={','.join(methods_to_run)}")
        except torch.cuda.OutOfMemoryError as exc:
            _write_all_error_rows(output, base, methods_to_run, "ERROR", exc)
            print(f"OOM shape_index={shape_index} m={m} n={n} k={k}: {exc}")
        except Exception as exc:
            _write_all_error_rows(output, base, methods_to_run, "ERROR", exc)
            print(f"ERROR shape_index={shape_index} m={m} n={n} k={k}: {type(exc).__name__}: {exc}")
        finally:
            _cleanup_cuda()


def _benchmark_shape(
    args: argparse.Namespace,
    output: str,
    base: dict[str, object],
    m: int,
    n: int,
    k: int,
    device: torch.device,
    methods_to_run: list[str],
    existing_shape: dict[str, dict[str, str]],
) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + int(base["shape_index"]))
    a_fp32 = torch.randn((m, k), device=device, dtype=torch.float32, generator=generator)
    w_fp32 = torch.randn((n, k), device=device, dtype=torch.float32, generator=generator)
    a_bf16 = a_fp32.to(torch.bfloat16).contiguous()
    w_bf16 = w_fp32.to(torch.bfloat16).contiguous()
    hdiag = a_fp32.pow(2).mean(dim=0)

    reference = F.linear(a_fp32, w_fp32)
    dense_fp32_ms = _existing_latency(existing_shape, "dense_fp32")
    dense_bf16_ms = _existing_latency(existing_shape, "dense_bf16")
    pending_rows: list[dict[str, object]] = []

    for method in methods_to_run:
        row = _run_method(args, base, method, a_fp32, w_fp32, a_bf16, w_bf16, hdiag, reference)
        if method == "dense_fp32" and row["status"] == "OK":
            dense_fp32_ms = float(row["latency_mean_ms"])
        if method == "dense_bf16" and row["status"] == "OK":
            dense_bf16_ms = float(row["latency_mean_ms"])
        pending_rows.append(row)

    for row in pending_rows:
        latency = _to_float(row.get("latency_mean_ms"))
        row["speedup_vs_dense_fp32"] = _format_ratio(dense_fp32_ms, latency)
        row["speedup_vs_dense_bf16"] = _format_ratio(dense_bf16_ms, latency)
        append_csv_row(output, CSV_FIELDS, row)


def _run_method(
    args: argparse.Namespace,
    base: dict[str, object],
    method: str,
    a_fp32: torch.Tensor,
    w_fp32: torch.Tensor,
    a_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
    hdiag: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    row = {
        **base,
        "method": method,
        "status": "OK",
        "error_type": "",
        "error_message": "",
        "compress_ms": "0.000000",
        "latency_mean_ms": "",
        "latency_p50_ms": "",
        "latency_min_ms": "",
        "latency_max_ms": "",
        "tflops": "",
        "mse": "",
        "abs_max_error": "",
        "reference_method": "dense_fp32",
        "speedup_vs_dense_fp32": "",
        "speedup_vs_dense_bf16": "",
        "kernel_backend": "",
        "prune_pattern": "",
        "actual_sparsity": "",
        "pad_multiple": "",
    }
    try:
        module, compress_ms, meta = _prepare_method(method, w_fp32, w_bf16, hdiag)
        row.update(meta)
        row["compress_ms"] = f"{compress_ms:.6f}"
        fn, out = _forward_fn(method, module, a_fp32, a_bf16, w_fp32, w_bf16)
        stats = _bench_cuda(fn, args.warmup, args.iters)
        diff = out.float() - reference.float()
        flops = float(row["flops"])
        row.update(
            {
                "latency_mean_ms": f"{stats.mean_ms:.6f}",
                "latency_p50_ms": f"{stats.p50_ms:.6f}",
                "latency_min_ms": f"{stats.min_ms:.6f}",
                "latency_max_ms": f"{stats.max_ms:.6f}",
                "tflops": f"{flops / (stats.mean_ms * 1.0e-3) / 1.0e12:.6f}",
                "mse": f"{diff.pow(2).mean().item():.8e}",
                "abs_max_error": f"{diff.abs().max().item():.8e}",
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        row.update(_error_fields("ERROR", exc))
        _cleanup_cuda()
    except RuntimeError as exc:
        status = "UNSUPPORTED" if _looks_unsupported(exc) else "ERROR"
        row.update(_error_fields(status, exc))
        _cleanup_cuda()
    except UNSUPPORTED_EXCEPTIONS as exc:
        row.update(_error_fields("UNSUPPORTED", exc))
        _cleanup_cuda()
    except Exception as exc:
        row.update(_error_fields("ERROR", exc))
        _cleanup_cuda()
    return row


def _prepare_method(
    method: str,
    w_fp32: torch.Tensor,
    w_bf16: torch.Tensor,
    hdiag: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | nn.Module | None, float, dict[str, object]]:
    if method == "dense_fp32":
        return None, 0.0, {"kernel_backend": "torch_dense_fp32"}
    if method == "dense_bf16":
        return None, 0.0, {"kernel_backend": "torch_dense_bf16"}
    if method == "dense_nvfp4":
        nvfp4_linear_cls, _can_use = _load_cutlass_nvfp4_symbols()
        linear = _linear_from_weight(w_bf16)
        compress_ms, module = _time_cuda(lambda: nvfp4_linear_cls.from_linear(linear).eval())
        return module, compress_ms, {"kernel_backend": "cutlass_nvfp4_sm120"}
    if method == "sparse_bf16":
        sparse_linear_cls, _can_use = _load_cutlass_sparse_bf16_symbols()
        compress_ms, prepared = _time_cuda(lambda: _prepare_sparse_bf16(sparse_linear_cls, w_bf16, hdiag))
        prune_result, sparse_linear = prepared
        module = PaddedSparseBF16Linear(sparse_linear, pad_multiple=8).eval()
        return module, compress_ms, _prune_meta("cutlass_sparse_bf16_cusparselt", prune_result, 8)
    if method == "sparse_nvfp4":
        sparse_linear_cls, _can_use = _load_cutlass_sparse_nvfp4_symbols()
        compress_ms, prepared = _time_cuda(lambda: _prepare_sparse_nvfp4(sparse_linear_cls, w_bf16, hdiag))
        prune_result, sparse_linear = prepared
        module = PaddedSparseNVFP4Linear(sparse_linear, pad_multiple=32).eval()
        return module, compress_ms, _prune_meta("cutlass_sparse_nvfp4_sm120", prune_result, 32)
    raise ValueError(f"Unsupported method: {method}")


def _prepare_sparse_bf16(
    sparse_linear_cls: type[nn.Module],
    w_bf16: torch.Tensor,
    hdiag: torch.Tensor,
) -> tuple[PruneResult, nn.Module]:
    prune_result = prune_dense_2_4(w_bf16, hdiag)
    _require_prune_ok(prune_result)
    linear = _linear_from_weight(prune_result.weight)
    return prune_result, sparse_linear_cls.from_linear(linear, prune=False).eval()


def _prepare_sparse_nvfp4(
    sparse_linear_cls: type[nn.Module],
    w_bf16: torch.Tensor,
    hdiag: torch.Tensor,
) -> tuple[PruneResult, nn.Module]:
    prune_result = prune_nvfp4_pair_2_4(w_bf16, hdiag)
    _require_prune_ok(prune_result)
    linear = _linear_from_weight(prune_result.weight)
    return prune_result, sparse_linear_cls.from_linear(linear, prune=False).eval()


def _forward_fn(
    method: str,
    module: Callable[[torch.Tensor], torch.Tensor] | nn.Module | None,
    a_fp32: torch.Tensor,
    a_bf16: torch.Tensor,
    w_fp32: torch.Tensor,
    w_bf16: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], torch.Tensor]:
    if method == "dense_fp32":
        fn = lambda: F.linear(a_fp32, w_fp32)
    elif method == "dense_bf16":
        fn = lambda: F.linear(a_bf16, w_bf16)
    else:
        assert module is not None
        fn = lambda: module(a_bf16)
    out = fn()
    torch.cuda.synchronize()
    return fn, out


def _linear_from_weight(weight: torch.Tensor) -> nn.Linear:
    out_features, in_features = int(weight.size(0)), int(weight.size(1))
    linear = nn.Linear(in_features, out_features, bias=False, device=weight.device, dtype=weight.dtype).eval()
    with torch.no_grad():
        linear.weight.copy_(weight)
    return linear


def _require_prune_ok(result: PruneResult) -> None:
    if result.mask is None:
        raise ValueError(_short_error_message(result.stats))


def _prune_meta(kernel_backend: str, result: PruneResult, pad_multiple: int) -> dict[str, object]:
    return {
        "kernel_backend": kernel_backend,
        "prune_pattern": result.stats.get("pattern", ""),
        "actual_sparsity": _format_float(result.stats.get("actual_sparsity", "")),
        "pad_multiple": pad_multiple,
    }


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
        times.append(float(start_event.elapsed_time(end_event)))
    sorted_times = sorted(times)
    mid = len(sorted_times) // 2
    if len(sorted_times) % 2 == 0:
        p50 = (sorted_times[mid - 1] + sorted_times[mid]) / 2.0
    else:
        p50 = sorted_times[mid]
    return BenchStats(
        mean_ms=sum(times) / len(times),
        p50_ms=p50,
        min_ms=min(times),
        max_ms=max(times),
    )


def _time_cuda(fn: Callable[[], object]) -> tuple[float, object]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize()
    return float(start_event.elapsed_time(end_event)), result


def _iter_shapes(
    fixed_dim: str,
    fixed_values: list[int],
    variable_values: list[int],
) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for fixed_value in fixed_values:
        for first in variable_values:
            for second in variable_values:
                if fixed_dim == "m":
                    shapes.append((fixed_value, first, second))
                elif fixed_dim == "n":
                    shapes.append((first, fixed_value, second))
                else:
                    shapes.append((first, second, fixed_value))
    return shapes


def _base_row(
    args: argparse.Namespace,
    output: str,
    shape_index: int,
    m: int,
    n: int,
    k: int,
    device: torch.device,
) -> dict[str, object]:
    fixed_value = {"m": m, "n": n, "k": k}[args.fixed_dim]
    flops = 2 * m * n * k
    dense_bf16_bytes = 2 * (m * k + n * k + m * n)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": "linear_kernel_shape_sweep",
        "output": output,
        "fixed_dim": args.fixed_dim,
        "fixed_value": fixed_value,
        "shape_index": shape_index,
        "m": m,
        "n": n,
        "k": k,
        "flops": flops,
        "dense_bf16_ai": _format_ratio(float(flops), float(dense_bf16_bytes)),
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_module": os.environ.get("CUDA_MODULE", ""),
    }


def _write_all_error_rows(
    output: str,
    base: dict[str, object],
    methods_to_run: list[str],
    status: str,
    exc: Exception,
) -> None:
    for method in methods_to_run:
        row = {
            **base,
            "method": method,
            **_blank_result_fields(),
            **_error_fields(status, exc),
        }
        append_csv_row(output, CSV_FIELDS, row)


def _load_existing_rows(output: str) -> dict[int, dict[str, dict[str, str]]]:
    path = Path(output)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    existing: dict[int, dict[str, dict[str, str]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            shape_index = _to_int(row.get("shape_index"))
            method = row.get("method", "")
            if shape_index < 0 or not method:
                continue
            existing.setdefault(shape_index, {})[method] = row
    return existing


def _existing_latency(existing_shape: dict[str, dict[str, str]], method: str) -> float | None:
    row = existing_shape.get(method)
    if not row or row.get("status") != "OK":
        return None
    return _to_float(row.get("latency_mean_ms"))


def _blank_result_fields() -> dict[str, object]:
    return {
        "compress_ms": "",
        "latency_mean_ms": "",
        "latency_p50_ms": "",
        "latency_min_ms": "",
        "latency_max_ms": "",
        "tflops": "",
        "mse": "",
        "abs_max_error": "",
        "reference_method": "dense_fp32",
        "speedup_vs_dense_fp32": "",
        "speedup_vs_dense_bf16": "",
        "kernel_backend": "",
        "prune_pattern": "",
        "actual_sparsity": "",
        "pad_multiple": "",
    }


def _error_fields(status: str, exc: Exception) -> dict[str, object]:
    return {
        "status": status,
        "error_type": type(exc).__name__,
        "error_message": _short_error_message(exc),
    }


def _looks_unsupported(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    unsupported_markers = (
        "cannot implement",
        "must be divisible",
        "shape_not_supported",
        "not divisible",
        "unsupported",
        "requires a cuda target device",
    )
    return any(marker in message for marker in unsupported_markers)


def _parse_int_list(value: str, name: str) -> list[int]:
    values = [int(part) for part in value.replace(" ", ",").split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers, got {values}")
    return values


def _to_float(value: object) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _to_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return -1


def _format_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return ""
    return f"{numerator / denominator:.6f}"


def _format_float(value: object) -> str:
    if value == "":
        return ""
    return f"{float(value):.6f}"


def _short_error_message(value: object, limit: int = 500) -> str:
    message = " ".join(str(value).split())
    return message[:limit]


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


CSV_FIELDS = [
    "timestamp",
    "task",
    "output",
    "fixed_dim",
    "fixed_value",
    "shape_index",
    "m",
    "n",
    "k",
    "flops",
    "dense_bf16_ai",
    "method",
    "status",
    "error_type",
    "error_message",
    "warmup",
    "iters",
    "seed",
    "compress_ms",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_min_ms",
    "latency_max_ms",
    "tflops",
    "mse",
    "abs_max_error",
    "reference_method",
    "speedup_vs_dense_fp32",
    "speedup_vs_dense_bf16",
    "kernel_backend",
    "prune_pattern",
    "actual_sparsity",
    "pad_multiple",
    "device",
    "torch_version",
    "cuda_version",
    "cuda_module",
]


if __name__ == "__main__":
    main()
