#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from common_kernel_prefill_loss import (
    DEBUG_ROOT,
    METHODS,
    SOURCE_ROOT,
    build_kernel_module_from_state,
    cleanup_cuda,
    compressible_modules,
    dtype_from_arg,
    load_calibration_blocks,
    load_llama_for_quality,
    load_prepared_payload,
    local_cuda_index,
    local_error_metadata,
    module_record,
    parse_methods,
    quality_config_from_args,
    select_modules,
    weight_stats,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-linear local errors through real NVFP4 runtime kernels.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--module-chunk-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    methods = parse_methods(args.methods)
    config = quality_config_from_args(args)
    blocks, calib_metadata = load_calibration_blocks(config)
    model = load_llama_for_quality(device=device, dtype=dtype)
    modules = select_modules(compressible_modules(model, "llama2-7b"), args.max_modules)
    states: dict[str, dict[str, torch.Tensor]] = {}
    source_metadata: dict[str, Any] = {}
    for method in methods:
        states[method], source_metadata[method] = load_prepared_payload(args.source_root, method)

    feature_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    chunk_size = max(args.module_chunk_size, 1)
    for chunk_start in range(0, len(modules), chunk_size):
        chunk = modules[chunk_start : chunk_start + chunk_size]
        print(f"collecting kernel local errors chunk {chunk_start + 1}-{chunk_start + len(chunk)} / {len(modules)}")
        accum = make_accum(chunk, methods)
        handles = []
        kernel_modules: dict[str, dict[str, torch.nn.Module]] = {method: {} for method in methods}
        try:
            for info in chunk:
                for method in methods:
                    kernel_modules[method][info.name] = build_kernel_module_from_state(
                        info.module,
                        name=info.name,
                        method=method,
                        state=states[method],
                        dtype=dtype,
                    )
                handles.append(info.module.register_forward_hook(make_hook(info.name, kernel_modules, accum)))
            loader = torch.utils.data.DataLoader(blocks, batch_size=args.batch_size, shuffle=False)
            for batch in loader:
                model(input_ids=batch.to(device=device, non_blocking=True), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
            del kernel_modules
            cleanup_cuda()

        for offset, info in enumerate(chunk, start=chunk_start + 1):
            base = module_record(info, offset)
            base.update(weight_stats(info.module))
            stats = accum[info.name]
            act_count = max(int(stats["activation_count"]), 1)
            act_mean = stats["activation_sum"] / act_count
            act_var = max(stats["activation_sq_sum"] / act_count - act_mean * act_mean, 0.0)
            base.update(
                {
                    "activation_mean": act_mean,
                    "activation_std": act_var**0.5,
                    "activation_abs_mean": stats["activation_abs_sum"] / act_count,
                    "activation_abs_max": stats["activation_abs_max"],
                    "activation_outlier_ratio_6x_weight_mean": stats["activation_outlier_count_6x_weight_mean"] / act_count,
                }
            )
            feature_rows.append(base)
            dense_weight = info.module.weight.detach().float().cpu()
            dense_weight_sq = max(float(dense_weight.pow(2).sum().item()), 1e-12)
            for method in methods:
                comp_weight = states[method][f"{info.name}.weight"].float()
                weight_err = (comp_weight - dense_weight).reshape(-1)
                weight_sse = float(weight_err.pow(2).sum().item())
                output_sse = stats["method_output_sse"][method]
                output_ref_sq = max(stats["method_output_ref_sq"][method], 1e-12)
                output_count = max(stats["method_output_count"][method], 1)
                weight_count = max(int(dense_weight.numel()), 1)
                error_rows.append(
                    {
                        **base,
                        "method": method,
                        "runtime_path": "real_kernel_forward",
                        "activation_quantization": "runtime_kernel",
                        "kernel_module_type": stats["method_kernel_module_type"][method],
                        "weight_mse": weight_sse / weight_count,
                        "weight_rel_mse": weight_sse / dense_weight_sq,
                        "weight_rmse_over_rms": (weight_sse / dense_weight_sq) ** 0.5,
                        "weight_max_abs_error": float(weight_err.abs().max().item()),
                        "output_mse": output_sse / output_count,
                        "output_rel_mse": output_sse / output_ref_sq,
                        "output_rmse_over_rms": (output_sse / output_ref_sq) ** 0.5,
                        "output_max_abs_error": stats["method_output_max_abs"][method],
                        "output_ref_rms": (output_ref_sq / output_count) ** 0.5,
                    }
                )
    write_csv(args.output_root / "sensitivity" / "module_features.csv", feature_rows)
    write_csv(args.output_root / "sensitivity" / "module_method_kernel_local_errors.csv", error_rows)
    write_json(
        args.output_root / "sensitivity" / "collect_kernel_local_errors_metadata.json",
        local_error_metadata(args, calib_metadata, methods, modules, source_metadata),
    )
    print(f"wrote {len(feature_rows)} feature rows and {len(error_rows)} kernel local-error rows")


def make_accum(modules: list[Any], methods: list[str]) -> dict[str, dict[str, Any]]:
    return {
        info.name: {
            "activation_sum": 0.0,
            "activation_sq_sum": 0.0,
            "activation_abs_sum": 0.0,
            "activation_abs_max": 0.0,
            "activation_count": 0,
            "activation_outlier_count_6x_weight_mean": 0,
            "method_output_sse": {method: 0.0 for method in methods},
            "method_output_ref_sq": {method: 0.0 for method in methods},
            "method_output_count": {method: 0 for method in methods},
            "method_output_max_abs": {method: 0.0 for method in methods},
            "method_kernel_module_type": {method: "" for method in methods},
        }
        for info in modules
    }


def make_hook(name: str, kernel_modules: dict[str, dict[str, torch.nn.Module]], accum: dict[str, dict[str, Any]]):
    def hook(module, inputs, output) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        x = inputs[0].detach()
        y_ref = output.detach()
        row = accum[name]
        flat_x = x.reshape(-1, x.shape[-1]).float()
        row["activation_sum"] += float(flat_x.sum().item())
        row["activation_sq_sum"] += float(flat_x.pow(2).sum().item())
        abs_x = flat_x.abs()
        row["activation_abs_sum"] += float(abs_x.sum().item())
        row["activation_abs_max"] = max(row["activation_abs_max"], float(abs_x.max().item()))
        row["activation_count"] += int(flat_x.numel())
        threshold = 6.0 * module.weight.detach().float().abs().mean().clamp(min=1e-12)
        row["activation_outlier_count_6x_weight_mean"] += int((abs_x > threshold).sum().item())
        ref_sq = float(y_ref.float().pow(2).sum().item())
        ref_count = int(y_ref.numel())
        for method, by_name in kernel_modules.items():
            kernel_module = by_name[name]
            row["method_kernel_module_type"][method] = type(kernel_module).__name__
            y_hat = kernel_module(x)
            err = (y_hat.float() - y_ref.float()).reshape(-1)
            row["method_output_sse"][method] += float(err.pow(2).sum().item())
            row["method_output_ref_sq"][method] += ref_sq
            row["method_output_count"][method] += ref_count
            row["method_output_max_abs"][method] = max(row["method_output_max_abs"][method], float(err.abs().max().item()))

    return hook


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        cleanup_cuda()
