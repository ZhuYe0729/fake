#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear
from fake.models.qwen3_5 import qwen3_5_model_path
from fake.models.qwen3_5_kernels import QwenHybridDenseNVFP4Linear


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/debug/001_qwen35_linear_breakdown/results"
DEFAULT_LAYERS = [
    "language_model.layers.0.linear_attn.in_proj_qkv",
    "language_model.layers.0.linear_attn.in_proj_z",
    "language_model.layers.0.mlp.gate_proj",
    "language_model.layers.0.mlp.down_proj",
    "language_model.layers.3.self_attn.q_proj",
    "language_model.layers.3.self_attn.o_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break down Qwen3.5 single-linear kernel timing.")
    parser.add_argument("--variant", default="9B")
    parser.add_argument("--layer", default="language_model.layers.0.mlp.down_proj")
    parser.add_argument("--layers", nargs="+", default=None, help="Measure multiple layers in one model load.")
    parser.add_argument("--default-w4a4-w4a16-layers", action="store_true", help="Measure representative Qwen3.5-9B layers whose predictor policy used dense_nvfp4/marlin_nvfp4.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.variant, dtype, device)
    layers = selected_layers(args)
    all_results = []
    for layer in layers:
        print(f"\n== Measuring {layer} ==")
        source_linear = resolve_module(model, layer)
        if not isinstance(source_linear, nn.Linear):
            raise TypeError(f"{layer} is {type(source_linear).__name__}, expected nn.Linear")
        base_linear = clone_linear(source_linear, device=device, dtype=dtype)
        results = measure_layer(args, layer, base_linear, device, dtype)
        layer_output_dir = output_dir_for_layer(args.output_dir, layer, multiple=len(layers) > 1)
        write_outputs(layer_output_dir, results)
        print_summary(results)
        all_results.append(results)
        del base_linear
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if len(all_results) > 1:
        write_aggregate_outputs(args.output_dir, all_results)


def selected_layers(args: argparse.Namespace) -> list[str]:
    if args.default_w4a4_w4a16_layers:
        return DEFAULT_LAYERS
    if args.layers:
        return list(args.layers)
    return [args.layer]


def output_dir_for_layer(base_output_dir: Path, layer: str, *, multiple: bool) -> Path:
    if not multiple:
        return base_output_dir
    return base_output_dir / sanitize_layer_name(layer)


def sanitize_layer_name(layer: str) -> str:
    return layer.replace("language_model.layers.0.", "").replace(".", "__")


def measure_layer(
    args: argparse.Namespace,
    layer: str,
    base_linear: nn.Linear,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    m_prefill = int(args.batch_size) * int(args.input_tokens)
    m_decode = int(args.batch_size)
    x_prefill = torch.randn((1, m_prefill, base_linear.in_features), device=device, dtype=dtype)
    x_decode = torch.randn((1, m_decode, base_linear.in_features), device=device, dtype=dtype)

    results = {
        "metadata": {
            "task": "qwen35_linear_breakdown",
            "model": f"Qwen3.5-{args.variant}",
            "model_path": str(qwen3_5_model_path(args.variant)),
            "layer": layer,
            "n": int(base_linear.out_features),
            "k": int(base_linear.in_features),
            "bias": base_linear.bias is not None,
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "batch_size": int(args.batch_size),
            "input_tokens": int(args.input_tokens),
            "output_tokens": int(args.output_tokens),
            "m_prefill": m_prefill,
            "m_decode": m_decode,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
        },
        "paths": {},
    }

    wrapper = load_wrapper()
    sparse = measure_sparse_bf16(wrapper, base_linear, x_prefill, x_decode, args)
    explicit_nvfp4 = measure_explicit_nvfp4(wrapper, base_linear, x_prefill, x_decode, args, dtype)
    lazy_nvfp4 = measure_lazy_nvfp4(wrapper, base_linear, x_prefill, x_decode, args, dtype)
    results["paths"] = {
        "sparse_bf16": sparse,
        "dense_nvfp4_prefill_marlin_decode_explicit": explicit_nvfp4,
        "dense_nvfp4_prefill_marlin_decode_lazy_wrapper": lazy_nvfp4,
    }
    return results


def load_model(variant: str, dtype: torch.dtype, device: torch.device) -> nn.Module:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(qwen3_5_model_path(variant)),
        trust_remote_code=True,
        dtype=dtype,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def resolve_module(model: nn.Module, path: str) -> nn.Module:
    current: Any = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def clone_linear(linear: nn.Linear, *, device: torch.device, dtype: torch.dtype) -> nn.Linear:
    cloned = nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=device,
        dtype=dtype,
    )
    cloned.weight.data.copy_(linear.weight.detach().to(device=device, dtype=dtype))
    if linear.bias is not None:
        cloned.bias.data.copy_(linear.bias.detach().to(device=device, dtype=dtype))
    cloned.eval()
    cloned.requires_grad_(False)
    return cloned


def measure_sparse_bf16(
    wrapper: Any,
    base_linear: nn.Linear,
    x_prefill: torch.Tensor,
    x_decode: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    build_ms, module = time_wall_cuda(
        lambda: PaddedSparseBF16Linear(
            wrapper.SparseBF16Linear.from_linear(base_linear, prune=True).eval(),
            8,
        ).eval()
    )
    prefill_first_ms = time_forward_once(module, x_prefill)
    prefill_steady_ms = time_forward_steady(module, x_prefill, args.warmup, args.iters)
    decode_first_ms = time_forward_once(module, x_decode)
    decode_steady_ms = time_forward_steady(module, x_decode, args.warmup, args.iters)
    return make_runtime_result(
        build_ms=build_ms,
        conversion_ms=0.0,
        prefill_first_ms=prefill_first_ms,
        prefill_steady_ms=prefill_steady_ms,
        decode_first_ms=decode_first_ms,
        decode_steady_ms=decode_steady_ms,
        output_tokens=args.output_tokens,
    )


def measure_explicit_nvfp4(
    wrapper: Any,
    base_linear: nn.Linear,
    x_prefill: torch.Tensor,
    x_decode: torch.Tensor,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, Any]:
    canonical_ms, canonical = time_wall_cuda(lambda: wrapper.canonical_from_linear(base_linear, device=base_linear.weight.device))
    cutlass_ms, cutlass_module = time_wall_cuda(
        lambda: wrapper.NVFP4Linear(wrapper.canonical_to_cutlass_nvfp4_weight(canonical)).eval()
    )
    prefill_first_ms = time_forward_once(cutlass_module, x_prefill)
    prefill_steady_ms = time_forward_steady(cutlass_module, x_prefill, args.warmup, args.iters)

    marlin_ms, marlin_module = time_wall_cuda(
        lambda: wrapper.MarlinNVFP4Linear(
            wrapper.canonical_to_marlin_nvfp4_weight(canonical, activation_dtype=dtype)
        ).eval()
    )
    decode_first_ms = time_forward_once(marlin_module, x_decode)
    decode_steady_ms = time_forward_steady(marlin_module, x_decode, args.warmup, args.iters)
    result = make_runtime_result(
        build_ms=canonical_ms,
        conversion_ms=cutlass_ms + marlin_ms,
        prefill_first_ms=prefill_first_ms,
        prefill_steady_ms=prefill_steady_ms,
        decode_first_ms=decode_first_ms,
        decode_steady_ms=decode_steady_ms,
        output_tokens=args.output_tokens,
    )
    result.update(
        {
            "canonical_build_ms": canonical_ms,
            "cutlass_materialize_ms": cutlass_ms,
            "marlin_materialize_ms": marlin_ms,
        }
    )
    return result


def measure_lazy_nvfp4(
    wrapper: Any,
    base_linear: nn.Linear,
    x_prefill: torch.Tensor,
    x_decode: torch.Tensor,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, Any]:
    canonical_ms, canonical = time_wall_cuda(lambda: wrapper.canonical_from_linear(base_linear, device=base_linear.weight.device))
    wrapper_build_ms, module = time_wall_cuda(
        lambda: QwenHybridDenseNVFP4Linear(
            canonical,
            decode_activation_dtype=dtype,
            marlin_m_threshold=int(args.batch_size),
            prefill_backend="dense_nvfp4",
            decode_backend="marlin_nvfp4",
        ).eval()
    )
    prefill_first_ms = time_forward_once(module, x_prefill)
    prefill_steady_ms = time_forward_steady(module, x_prefill, args.warmup, args.iters)
    decode_first_ms = time_forward_once(module, x_decode)
    decode_steady_ms = time_forward_steady(module, x_decode, args.warmup, args.iters)
    result = make_runtime_result(
        build_ms=canonical_ms + wrapper_build_ms,
        conversion_ms=0.0,
        prefill_first_ms=prefill_first_ms,
        prefill_steady_ms=prefill_steady_ms,
        decode_first_ms=decode_first_ms,
        decode_steady_ms=decode_steady_ms,
        output_tokens=args.output_tokens,
    )
    result.update(
        {
            "canonical_build_ms": canonical_ms,
            "wrapper_build_ms": wrapper_build_ms,
            "lazy_prefill_materialization_in_first_ms": prefill_first_ms,
            "lazy_decode_materialization_in_first_ms": decode_first_ms,
        }
    )
    return result


def make_runtime_result(
    *,
    build_ms: float,
    conversion_ms: float,
    prefill_first_ms: float,
    prefill_steady_ms: float,
    decode_first_ms: float,
    decode_steady_ms: float,
    output_tokens: int,
) -> dict[str, Any]:
    decode_x_n_steady_ms = int(output_tokens) * decode_steady_ms
    runtime_only_steady_ms = prefill_steady_ms + decode_x_n_steady_ms
    runtime_first_inclusive_ms = prefill_first_ms + decode_first_ms + max(int(output_tokens) - 1, 0) * decode_steady_ms
    return {
        "build_ms": build_ms,
        "conversion_ms": conversion_ms,
        "prefill_first_ms": prefill_first_ms,
        "prefill_steady_ms": prefill_steady_ms,
        "decode_first_ms": decode_first_ms,
        "decode_steady_ms": decode_steady_ms,
        "decode_x_n_steady_ms": decode_x_n_steady_ms,
        "runtime_only_steady_ms": runtime_only_steady_ms,
        "runtime_first_inclusive_ms": runtime_first_inclusive_ms,
        "e2e_with_build_or_conversion_steady_ms": build_ms + conversion_ms + runtime_only_steady_ms,
        "e2e_with_build_or_conversion_first_inclusive_ms": build_ms + conversion_ms + runtime_first_inclusive_ms,
    }


def time_wall_cuda(fn: Callable[[], Any]) -> tuple[float, Any]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, result


def time_forward_once(module: nn.Module, x: torch.Tensor) -> float:
    latency_ms, result = time_cuda_event(lambda: module(x))
    assert_finite(result)
    del result
    return latency_ms


def time_forward_steady(module: nn.Module, x: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        result = module(x)
        assert_finite(result)
        del result
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        latency_ms, result = time_cuda_event(lambda: module(x))
        assert_finite(result)
        del result
        times.append(latency_ms)
    return sum(times) / len(times)


def time_cuda_event(fn: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), result


def assert_finite(tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all().item():
        raise RuntimeError("Output contains NaN/Inf")


def load_wrapper() -> Any:
    import importlib

    for module_name in ("fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper", "cutlass_wrapper"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass
    raise RuntimeError("CUTLASS wrapper package is not importable")


def write_outputs(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "breakdown.json"
    csv_path = output_dir / "breakdown.csv"
    readme_path = output_dir.parent / "README.md" if output_dir.name == "results" else output_dir / "README.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    rows = flatten_results(results)
    with csv_path.open("w", newline="") as f:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    readme_path.write_text(render_readme(results))


def write_aggregate_outputs(output_dir: Path, all_results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for results in all_results:
        rows.extend(flatten_results(results))
    csv_path = output_dir / "aggregate_breakdown.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "aggregate_breakdown.json"
    json_path.write_text(json.dumps(all_results, indent=2) + "\n")
    readme_path = output_dir.parent / "README.md"
    readme_path.write_text(render_aggregate_readme(all_results))


def flatten_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = results["metadata"]
    rows = []
    for path_name, values in results["paths"].items():
        row = {
            "model": metadata["model"],
            "layer": metadata["layer"],
            "n": metadata["n"],
            "k": metadata["k"],
            "batch_size": metadata["batch_size"],
            "input_tokens": metadata["input_tokens"],
            "output_tokens": metadata["output_tokens"],
            "m_prefill": metadata["m_prefill"],
            "m_decode": metadata["m_decode"],
            "path": path_name,
        }
        row.update({key: format_float(value) if isinstance(value, float) else value for key, value in values.items()})
        rows.append(row)
    return rows


def render_readme(results: dict[str, Any]) -> str:
    metadata = results["metadata"]
    lines = [
        "# Qwen3.5-9B Linear Kernel Breakdown",
        "",
        "## Scenario",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Layer: `{metadata['layer']}`",
        f"- Shape: `N={metadata['n']}, K={metadata['k']}`",
        f"- Workload: `batch_size={metadata['batch_size']}, input_tokens={metadata['input_tokens']}, output_tokens={metadata['output_tokens']}`",
        f"- M: `prefill={metadata['m_prefill']}, decode={metadata['m_decode']}`",
        f"- GPU: `{metadata['gpu_name']}`",
        "",
        "## Breakdown",
        "",
        "| Path | Build ms | Conversion ms | Prefill first ms | Prefill steady ms | Decode first ms | Decode steady ms | Decode x32 steady ms | Runtime steady ms | E2E steady with build/conversion ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path_name, values in results["paths"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{path_name}`",
                    format_float(values["build_ms"]),
                    format_float(values["conversion_ms"]),
                    format_float(values["prefill_first_ms"]),
                    format_float(values["prefill_steady_ms"]),
                    format_float(values["decode_first_ms"]),
                    format_float(values["decode_steady_ms"]),
                    format_float(values["decode_x_n_steady_ms"]),
                    format_float(values["runtime_only_steady_ms"]),
                    format_float(values["e2e_with_build_or_conversion_steady_ms"]),
                ]
            )
            + " |"
        )
    sparse = results["paths"].get("sparse_bf16")
    explicit = results["paths"].get("dense_nvfp4_prefill_marlin_decode_explicit")
    lazy = results["paths"].get("dense_nvfp4_prefill_marlin_decode_lazy_wrapper")
    if sparse and explicit and lazy:
        runtime_gap = sparse["runtime_only_steady_ms"] - explicit["runtime_only_steady_ms"]
        sparse_build = sparse["build_ms"]
        explicit_conversion = explicit["build_ms"] + explicit["conversion_ms"]
        lazy_first_extra = lazy["runtime_first_inclusive_ms"] - lazy["runtime_only_steady_ms"]
        lines.extend(
            [
                "",
                "## Observations",
                "",
                f"- Runtime-only steady latency: sparse_bf16 is `{format_float(sparse['runtime_only_steady_ms'])}ms`, explicit dense_nvfp4+marlin is `{format_float(explicit['runtime_only_steady_ms'])}ms`; sparse is `{format_float(runtime_gap)}ms` slower for this layer and workload.",
                f"- Offline preparation is very different: sparse_bf16 build/pack is `{format_float(sparse_build)}ms`, while explicit canonical+CUTLASS+Marlin preparation is `{format_float(explicit_conversion)}ms` in this run.",
                f"- Lazy wrapper steady runtime is close to explicit dense_nvfp4+marlin, but the first prefill/decode calls include lazy materialization; first-inclusive runtime is `{format_float(lazy['runtime_first_inclusive_ms'])}ms`, `{format_float(lazy_first_extra)}ms` above steady runtime.",
                "- These numbers are single-layer debug timings. Build/materialization costs are offline costs unless the wrapper leaves them lazy and pays them during the first timed forward.",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `first` includes any first-call runtime initialization that remains after explicit build/materialization.",
            "- The lazy wrapper row intentionally leaves CUTLASS/Marlin materialization inside the first prefill/decode calls.",
            "- `runtime steady` excludes build/conversion and uses warmed forward latency.",
            "- `E2E steady with build/conversion` adds build/conversion once to steady prefill + 32 decode steps.",
            "",
            "## Files",
            "",
            "- `results/breakdown.json`: full structured result.",
            "- `results/breakdown.csv`: flat comparison table.",
            "- `scripts/qwen35_linear_breakdown.py`: reproduction script.",
            "",
        ]
    )
    return "\n".join(lines)


def render_aggregate_readme(all_results: list[dict[str, Any]]) -> str:
    first = all_results[0]["metadata"]
    lines = [
        "# Qwen3.5-9B Multi-Linear Kernel Breakdown",
        "",
        "## Scenario",
        "",
        f"- Model: `{first['model']}`",
        f"- Workload: `batch_size={first['batch_size']}, input_tokens={first['input_tokens']}, output_tokens={first['output_tokens']}`",
        f"- M: `prefill={first['m_prefill']}, decode={first['m_decode']}`",
        f"- GPU: `{first['gpu_name']}`",
        "",
        "## Runtime-Only Steady Summary",
        "",
        "| Layer | Shape | sparse_bf16 ms | explicit dense_nvfp4+marlin ms | lazy wrapper ms | sparse - explicit ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for results in all_results:
        metadata = results["metadata"]
        sparse = results["paths"]["sparse_bf16"]["runtime_only_steady_ms"]
        explicit = results["paths"]["dense_nvfp4_prefill_marlin_decode_explicit"]["runtime_only_steady_ms"]
        lazy = results["paths"]["dense_nvfp4_prefill_marlin_decode_lazy_wrapper"]["runtime_only_steady_ms"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{metadata['layer']}`",
                    f"`N={metadata['n']},K={metadata['k']}`",
                    format_float(sparse),
                    format_float(explicit),
                    format_float(lazy),
                    format_float(sparse - explicit),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Prefill/Decode Detail",
            "",
            "| Layer | Path | Prefill steady ms | Decode steady ms | Decode x32 ms | Runtime steady ms | Build+conversion ms |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for results in all_results:
        metadata = results["metadata"]
        for path_name, values in results["paths"].items():
            build_conversion = values["build_ms"] + values["conversion_ms"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{metadata['layer']}`",
                        f"`{path_name}`",
                        format_float(values["prefill_steady_ms"]),
                        format_float(values["decode_steady_ms"]),
                        format_float(values["decode_x_n_steady_ms"]),
                        format_float(values["runtime_only_steady_ms"]),
                        format_float(build_conversion),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This debug run targets representative Qwen3.5-9B layers whose predictor policy used `dense_nvfp4/marlin_nvfp4`.",
            "- `runtime steady` excludes offline build/conversion and uses warmed forward latency.",
            "- Per-layer subdirectories contain full JSON/CSV breakdowns.",
            "",
            "## Files",
            "",
            "- `results/aggregate_breakdown.csv`: flat multi-layer table.",
            "- `results/aggregate_breakdown.json`: structured multi-layer data.",
            "- `results/<layer>/breakdown.csv`: per-layer detail.",
            "",
        ]
    )
    return "\n".join(lines)


def print_summary(results: dict[str, Any]) -> None:
    metadata = results["metadata"]
    print(
        f"Qwen3.5 breakdown complete: layer={metadata['layer']} "
        f"N={metadata['n']} K={metadata['k']} "
        f"M_prefill={metadata['m_prefill']} M_decode={metadata['m_decode']}"
    )
    for path_name, values in results["paths"].items():
        print(
            f"  {path_name}: runtime={values['runtime_only_steady_ms']:.4f}ms "
            f"with_build={values['e2e_with_build_or_conversion_steady_ms']:.4f}ms "
            f"prefill={values['prefill_steady_ms']:.4f}ms "
            f"decode_step={values['decode_steady_ms']:.4f}ms"
        )


def format_float(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
