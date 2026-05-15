#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from math import prod
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fake.kernels.flashinfer_nvfp4 import (
    FlashInferNVFP4Config,
    FlashInferNVFP4Linear,
    flashinfer_version,
)
from fake.kernels.flashinfer_nvfp4 import _global_scale as nvfp4_global_scale
from fake.models.dinov3 import DEFAULT_DINOV3_BACKBONE_PATH, DEFAULT_DINOV3_HEAD_PATH, model_input_dtype
from fake.models.dinov3_nvfp4 import load_dinov3_vit7b16_flashinfer_nvfp4_classifier
from fake.utils.csv_io import append_csv_row


DEFAULT_INPUT_SIZES = ("3x128x128", "3x256x256", "3x384x384")


@dataclass(frozen=True)
class BenchStats:
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class LayerCall:
    name: str
    call_index: int
    input_shape: tuple[int, ...]
    m: int
    k: int
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DINOv3 ViT-7B/16 FlashInfer NVFP4 Linear layers and primitive costs."
    )
    parser.add_argument("--backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--input-sizes", nargs="+", default=list(DEFAULT_INPUT_SIZES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--output", default="artifacts/analysis/dinov3_vit7b16/nvfp4/microbench.csv")
    parser.add_argument("--gemm-backend", choices=["auto", "cutlass", "cudnn", "trtllm", "cute-dsl", "b12x"], default="auto")
    parser.add_argument("--quant-backend", choices=["cuda", "cute-dsl"], default="cuda")
    parser.add_argument("--out-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--sf-layout", choices=["layout_128x4", "layout_8x4", "layout_linear"], default="layout_128x4")
    parser.add_argument("--per-token-activation", action="store_true")
    parser.add_argument("--fallback-on-error", action="store_true")
    parser.add_argument("--max-layers", type=int, default=None, help="Optional quick-run limit for layer calls per input size.")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be >= 0 and --iters must be > 0")
    batch_sizes = args.batch_sizes if args.batch_sizes is not None else [args.batch_size]
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("--batch-size and --batch-sizes values must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    nvfp4_config = FlashInferNVFP4Config(
        sf_layout=args.sf_layout,
        gemm_backend=args.gemm_backend,
        quant_backend=args.quant_backend,
        out_dtype=args.out_dtype,
        per_token_activation=args.per_token_activation,
        fallback_on_error=args.fallback_on_error,
    )
    model, config, report = load_dinov3_vit7b16_flashinfer_nvfp4_classifier(
        backbone_path=args.backbone_path,
        head_path=args.head_path,
        device=device,
        dtype=args.dtype,
        nvfp4_config=nvfp4_config,
    )
    input_dtype = model_input_dtype(model)
    modules = dict(_iter_nvfp4_modules(model))
    if not modules:
        raise RuntimeError("No FlashInferNVFP4Linear modules were found in the converted model.")

    print(
        "dinov3 vit7b16 nvfp4 microbench: "
        f"layers={len(modules)} replaced={report.replaced_linear_count} "
        f"skipped={report.skipped_linear_count} output={args.output}"
    )

    input_sizes = [_parse_input_size(value) for value in args.input_sizes]
    for batch_size in batch_sizes:
        for input_size in input_sizes:
            base = _base_row(
                args,
                config,
                batch_size,
                input_size,
                input_dtype,
                device,
                report.replaced_linear_count,
                report.skipped_linear_count,
            )
            try:
                _validate_dinov3_input_size(input_size)
                calls = _capture_layer_calls(model, modules, batch_size, input_size, input_dtype, device)
                if args.max_layers is not None:
                    calls = calls[: args.max_layers]

                model_input = torch.randn((batch_size, *input_size), device=device, dtype=input_dtype)
                model_stats = _bench_cuda(lambda: model(model_input), args.warmup, args.iters)
                _write_row(
                    args.output,
                    base,
                    {
                        "status": "OK",
                        "error_type": "",
                        "error_message": "",
                        "layer_name": "__model__",
                        "call_index": 0,
                        "input_shape": _shape_str(tuple(model_input.shape)),
                        "m": "",
                        "k": "",
                        "n": "",
                        "op": "model_forward",
                        **_stats_fields(model_stats),
                    },
                )

                unique_calls = _dedupe_layer_calls(calls)
                summaries: dict[str, float] = {}
                for call in unique_calls:
                    module = modules[call.name]
                    layer_rows = _benchmark_layer_call(module, call, args.warmup, args.iters, input_dtype, device)
                    for op, stats in layer_rows:
                        summaries[op] = summaries.get(op, 0.0) + stats.mean_ms
                        _write_row(
                            args.output,
                            base,
                            {
                                "status": "OK",
                                "error_type": "",
                                "error_message": "",
                                "layer_name": call.name,
                                "call_index": call.call_index,
                                "input_shape": _shape_str(call.input_shape),
                                "m": call.m,
                                "k": call.k,
                                "n": call.n,
                                "op": op,
                                **_stats_fields(stats),
                            },
                        )

                print(
                    f"batch_size={batch_size} input_size="
                    f"{_shape_str(input_size)} model_forward_ms={model_stats.mean_ms:.6f} "
                    f"sum_layer_forward_ms={summaries.get('layer_forward', 0.0):.6f} "
                    f"sum_gemm_only_ms={summaries.get('gemm_only', 0.0):.6f} "
                    f"sum_activation_quant_only_ms={summaries.get('activation_quant_only', 0.0):.6f} "
                    f"unique_layer_calls={len(unique_calls)} raw_layer_calls={len(calls)}"
                )
            except Exception as exc:
                _write_error_row(args.output, base, exc)
                _cleanup_cuda()
                print(f"ERROR batch_size={batch_size} input_size={_shape_str(input_size)}: {type(exc).__name__}: {exc}")

    if report.skipped:
        print(f"skipped_modules={report.skipped[:10]}")


def _iter_nvfp4_modules(model: torch.nn.Module) -> list[tuple[str, FlashInferNVFP4Linear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, FlashInferNVFP4Linear)]


def _dedupe_layer_calls(calls: list[LayerCall]) -> list[LayerCall]:
    seen: set[tuple[tuple[int, ...], int, int, int]] = set()
    unique_calls: list[LayerCall] = []
    for call in calls:
        key = (call.input_shape, call.m, call.n, call.k)
        if key in seen:
            continue
        seen.add(key)
        unique_calls.append(call)
    return unique_calls


def _capture_layer_calls(
    model: torch.nn.Module,
    modules: dict[str, FlashInferNVFP4Linear],
    batch_size: int,
    input_size: tuple[int, int, int],
    input_dtype: torch.dtype,
    device: torch.device,
) -> list[LayerCall]:
    calls: list[LayerCall] = []
    seen: dict[str, int] = {}
    handles = []

    def make_hook(name: str) -> Callable:
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            x = inputs[0]
            call_index = seen.get(name, 0)
            seen[name] = call_index + 1
            input_shape = tuple(int(dim) for dim in x.shape)
            calls.append(
                LayerCall(
                    name=name,
                    call_index=call_index,
                    input_shape=input_shape,
                    m=int(prod(input_shape[:-1])),
                    k=int(input_shape[-1]),
                    n=int(modules[name].out_features),
                )
            )

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        sample = torch.randn((batch_size, *input_size), device=device, dtype=input_dtype)
        model(sample)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    return calls


def _benchmark_layer_call(
    module: FlashInferNVFP4Linear,
    call: LayerCall,
    warmup: int,
    iters: int,
    input_dtype: torch.dtype,
    device: torch.device,
) -> list[tuple[str, BenchStats]]:
    x = torch.randn(call.input_shape, device=device, dtype=input_dtype)
    x_2d = x.reshape(-1, call.k).contiguous()
    activation_global_sf = nvfp4_global_scale(x_2d)
    quant_result = _activation_quant(module, x_2d, activation_global_sf)
    activation_fp4, activation_sf, alpha = _unpack_activation_quant(module, quant_result, activation_global_sf)

    rows = [
        ("layer_forward", _bench_cuda(lambda: module(x), warmup, iters)),
        ("forward_like_2d", _bench_cuda(lambda: _forward_like_2d(module, x_2d), warmup, iters)),
        ("activation_global_scale", _bench_cuda(lambda: nvfp4_global_scale(x_2d), warmup, iters)),
        (
            "activation_quant_only",
            _bench_cuda(lambda: _activation_quant(module, x_2d, activation_global_sf), warmup, iters),
        ),
        (
            "activation_scale_plus_quant",
            _bench_cuda(lambda: _activation_scale_plus_quant(module, x_2d), warmup, iters),
        ),
        ("alpha", _bench_cuda(lambda: _alpha(module, activation_global_sf, quant_result), warmup, iters)),
        ("gemm_only", _bench_cuda(lambda: _gemm(module, activation_fp4, activation_sf, alpha), warmup, iters)),
    ]
    if module.bias is not None:
        gemm_out = _gemm(module, activation_fp4, activation_sf, alpha)
        rows.append(("bias_add", _bench_cuda(lambda: gemm_out + module.bias.to(dtype=gemm_out.dtype), warmup, iters)))
    rows.append(
        (
            "dense_linear",
            _bench_cuda(lambda: F.linear(x_2d, module.fallback_weight, module.bias), warmup, iters),
        )
    )
    return rows


def _forward_like_2d(module: FlashInferNVFP4Linear, x_2d: torch.Tensor) -> torch.Tensor:
    activation_global_sf = nvfp4_global_scale(x_2d)
    quant_result = _activation_quant(module, x_2d, activation_global_sf)
    activation_fp4, activation_sf, alpha = _unpack_activation_quant(module, quant_result, activation_global_sf)
    out = _gemm(module, activation_fp4, activation_sf, alpha)
    if module.bias is not None:
        out = out + module.bias.to(dtype=out.dtype)
    return out


def _activation_scale_plus_quant(module: FlashInferNVFP4Linear, x_2d: torch.Tensor):
    activation_global_sf = nvfp4_global_scale(x_2d)
    return _activation_quant(module, x_2d, activation_global_sf)


def _activation_quant(
    module: FlashInferNVFP4Linear,
    x_2d: torch.Tensor,
    activation_global_sf: torch.Tensor,
):
    return module._flashinfer.nvfp4_quantize(
        x_2d,
        activation_global_sf,
        sfLayout=module._sf_layout,
        do_shuffle=False,
        sf_vec_size=module.config.block_size,
        backend=module.config.quant_backend,
        per_token_activation=module.config.per_token_activation,
    )


def _unpack_activation_quant(
    module: FlashInferNVFP4Linear,
    quant_result,
    activation_global_sf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if module.config.per_token_activation:
        activation_fp4, activation_sf, per_token_sf = quant_result
        alpha = per_token_sf.reshape(-1, 1) / module.weight_global_sf
        return activation_fp4, activation_sf, alpha
    activation_fp4, activation_sf = quant_result
    alpha = torch.reciprocal(activation_global_sf * module.weight_global_sf)
    return activation_fp4, activation_sf, alpha


def _alpha(module: FlashInferNVFP4Linear, activation_global_sf: torch.Tensor, quant_result) -> torch.Tensor:
    if module.config.per_token_activation:
        _activation_fp4, _activation_sf, per_token_sf = quant_result
        return per_token_sf.reshape(-1, 1) / module.weight_global_sf
    return torch.reciprocal(activation_global_sf * module.weight_global_sf)


def _gemm(
    module: FlashInferNVFP4Linear,
    activation_fp4: torch.Tensor,
    activation_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return module._flashinfer.mm_fp4(
        activation_fp4,
        module.weight_fp4_t,
        activation_sf,
        module.weight_sf_t,
        alpha,
        module._out_dtype,
        None,
        module.config.block_size,
        module.config.sf_layout == "layout_8x4",
        module.config.gemm_backend,
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


def _base_row(
    args: argparse.Namespace,
    config: dict,
    batch_size: int,
    input_size: tuple[int, int, int],
    input_dtype: torch.dtype,
    device: torch.device,
    replaced_linear_count: int,
    skipped_linear_count: int,
) -> dict[str, object]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "head": "dinov3_vit7b16_imagenet1k_linear_head",
        "method": "nvfp4_flashinfer",
        "task": "microbench",
        "dtype_arg": args.dtype,
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "input_c": input_size[0],
        "input_h": input_size[1],
        "input_w": input_size[2],
        "warmup": args.warmup,
        "iters": args.iters,
        "hidden_size": config.get("hidden_size", ""),
        "num_hidden_layers": config.get("num_hidden_layers", ""),
        "num_register_tokens": config.get("num_register_tokens", ""),
        "backbone_path": args.backbone_path,
        "head_path": args.head_path,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_module": os.environ.get("CUDA_MODULE", ""),
        "flashinfer_version": flashinfer_version(),
        "nvfp4_block_size": 16,
        "nvfp4_gemm_backend": args.gemm_backend,
        "nvfp4_quant_backend": args.quant_backend,
        "nvfp4_sf_layout": args.sf_layout,
        "nvfp4_per_token_activation": args.per_token_activation,
        "replaced_linear_count": replaced_linear_count,
        "skipped_linear_count": skipped_linear_count,
    }


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
            "layer_name": "__error__",
            "call_index": "",
            "input_shape": "",
            "m": "",
            "k": "",
            "n": "",
            "op": "error",
            "latency_mean_ms": "",
            "latency_min_ms": "",
            "latency_max_ms": "",
        },
    )


def _short_error_message(exc: Exception, limit: int = 500) -> str:
    message = " ".join(str(exc).split())
    return message[:limit]


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stats_fields(stats: BenchStats) -> dict[str, str]:
    return {
        "latency_mean_ms": f"{stats.mean_ms:.6f}",
        "latency_min_ms": f"{stats.min_ms:.6f}",
        "latency_max_ms": f"{stats.max_ms:.6f}",
    }


def _parse_input_size(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected CxHxW input size, got: {value}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _validate_dinov3_input_size(input_size: tuple[int, int, int]) -> None:
    channels, height, width = input_size
    if channels != 3:
        raise ValueError(f"DINOv3 expects 3 input channels, got input_size={_shape_str(input_size)}")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            "DINOv3 ViT-7B/16 expects input height/width to be divisible by patch size 16; "
            f"got input_size={_shape_str(input_size)}."
        )


def _shape_str(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


if __name__ == "__main__":
    main()
