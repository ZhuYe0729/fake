#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common_quality import (
    COMPRESSED_CORE_METHODS,
    DEBUG_ROOT,
    MODEL_KEY,
    QualityConfig,
    cleanup_cuda,
    compressible_modules,
    compute_nll,
    dtype_from_arg,
    layer_index,
    load_calibration_blocks,
    load_llama_for_quality,
    load_prepared_state,
    local_cuda_index,
    module_record,
    tensor_stats,
    weight_stats,
    write_csv,
    write_json,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Llama2 module-level quality proxy features.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEBUG_ROOT.parents[1] / "results/main/003_llama2_7b_arc_easy_accuracy")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--module-chunk-size", type=int, default=8)
    parser.add_argument("--methods", default=",".join(COMPRESSED_CORE_METHODS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    config = QualityConfig(
        calib_samples=args.calib_samples,
        seq_len=args.seq_len,
        seed=args.seed,
        batch_size=args.batch_size,
        source_root=args.source_root,
        output_root=args.output_root,
    )
    blocks, calib_metadata = load_calibration_blocks(config)
    write_json(args.output_root / "calibration" / "wikitext2_quality_metadata.json", calib_metadata)

    model = load_llama_for_quality(device=device, dtype=dtype)
    modules = compressible_modules(model, MODEL_KEY)
    if args.max_modules is not None:
        modules = modules[: args.max_modules]
    module_names = [info.name for info in modules]
    compressed_states = {method: load_prepared_state(args.source_root, method) for method in methods}

    dense_quality = compute_nll(model, blocks, device=device, batch_size=args.batch_size)
    write_json(args.output_root / "sensitivity" / "dense_nll.json", dense_quality)

    feature_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    def make_hook(name: str, method_weights: dict[str, torch.Tensor], accum: dict[str, dict[str, Any]]):

        def hook(module, inputs, output) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            x = inputs[0].detach()
            y_ref = output.detach()
            flat = x.reshape(-1, x.shape[-1]).float()
            row = accum[name]
            row["activation_sum"] += float(flat.sum().item())
            row["activation_sq_sum"] += float(flat.pow(2).sum().item())
            abs_flat = flat.abs()
            row["activation_abs_sum"] += float(abs_flat.sum().item())
            row["activation_abs_max"] = max(row["activation_abs_max"], float(abs_flat.max().item()))
            row["activation_count"] += int(flat.numel())
            threshold = 6.0 * module.weight.detach().float().abs().mean().clamp(min=1e-12)
            row["activation_outlier_count_6x_weight_mean"] += int((abs_flat > threshold).sum().item())
            ref_sq = float(y_ref.float().pow(2).sum().item())
            for method, weight in method_weights.items():
                y_hat = F.linear(x, weight, module.bias)
                err = (y_hat.float() - y_ref.float()).reshape(-1)
                row["method_sse"][method] += float(err.pow(2).sum().item())
                row["method_ref_sq"][method] += ref_sq
                row["method_max_abs"][method] = max(row["method_max_abs"][method], float(err.abs().max().item()))

        return hook

    chunk_size = max(args.module_chunk_size, 1)
    for chunk_start in range(0, len(modules), chunk_size):
        chunk = modules[chunk_start : chunk_start + chunk_size]
        print(f"collecting chunk {chunk_start + 1}-{chunk_start + len(chunk)} / {len(modules)}")
        accum = {
            info.name: {
                "activation_sum": 0.0,
                "activation_sq_sum": 0.0,
                "activation_abs_sum": 0.0,
                "activation_abs_max": 0.0,
                "activation_count": 0,
                "activation_outlier_count_6x_weight_mean": 0,
                "method_sse": {method: 0.0 for method in methods},
                "method_ref_sq": {method: 0.0 for method in methods},
                "method_max_abs": {method: 0.0 for method in methods},
            }
            for info in chunk
        }
        handles = []
        try:
            for info in chunk:
                method_weights = {
                    method: compressed_states[method][f"{info.name}.weight"].to(device=device, dtype=dtype)
                    for method in methods
                }
                handles.append(info.module.register_forward_hook(make_hook(info.name, method_weights, accum)))
            loader = torch.utils.data.DataLoader(blocks, batch_size=args.batch_size, shuffle=False)
            for batch in loader:
                model(input_ids=batch.to(device=device, non_blocking=True), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
            cleanup_cuda()

        for offset, info in enumerate(chunk, start=chunk_start + 1):
            base = module_record(info, offset)
            base.update(weight_stats(info.module))
            stats = accum[info.name]
            count = max(int(stats["activation_count"]), 1)
            act_mean = stats["activation_sum"] / count
            act_var = max(stats["activation_sq_sum"] / count - act_mean * act_mean, 0.0)
            act_abs_mean = stats["activation_abs_sum"] / count
            base.update(
                {
                    "activation_mean": act_mean,
                    "activation_std": act_var**0.5,
                    "activation_abs_mean": act_abs_mean,
                    "activation_abs_max": stats["activation_abs_max"],
                    "activation_outlier_ratio_6x_weight_mean": stats["activation_outlier_count_6x_weight_mean"] / count,
                }
            )
            feature_rows.append(base)
            for method in methods:
                sse = stats["method_sse"][method]
                ref_sq = max(stats["method_ref_sq"][method], 1e-12)
                error_rows.append(
                    {
                        **base,
                        "method": method,
                        "local_rel_mse": sse / ref_sq,
                        "local_rmse_over_rms": (sse / ref_sq) ** 0.5,
                        "local_max_abs_error": stats["method_max_abs"][method],
                    }
                )

    write_csv(args.output_root / "sensitivity" / "module_features.csv", feature_rows)
    write_csv(args.output_root / "sensitivity" / "module_method_errors.csv", error_rows)
    write_run_metadata(
        args.output_root / "sensitivity" / "collect_sensitivity_metadata.json",
        {
            "source_root": str(args.source_root),
            "dtype": args.dtype,
            "gpu": args.gpu,
            "methods": methods,
            "calibration": calib_metadata,
            "dense_quality": dense_quality,
            "selected_modules": len(modules),
        },
    )
    print(f"wrote {len(feature_rows)} feature rows and {len(error_rows)} method-error rows")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda()
