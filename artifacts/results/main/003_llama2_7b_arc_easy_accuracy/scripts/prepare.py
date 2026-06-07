#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from common import (
    COMPRESSED_METHODS,
    DEFAULT_MODEL_KEY,
    EXPERIMENT_ROOT,
    CalibConfig,
    CompressionConfig,
    append_jsonl,
    build_calib_loader,
    build_wikitext2_blocks,
    cleanup_cuda,
    collect_hessian_diag_llama,
    collect_hessians_for_modules_llama,
    compressible_modules,
    cpu_state_dict,
    dtype_from_arg,
    load_model,
    model_result_root,
    model_spec,
    module_parent,
    nvfp4_quantize_weight,
    sparsegpt_prune_weight,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real compressed Llama-2-7B artifacts.")
    parser.add_argument("--method", choices=COMPRESSED_METHODS, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--sparsegpt-block-size", type=int, default=128)
    parser.add_argument("--sparsegpt-percdamp", type=float, default=0.01)
    parser.add_argument("--nvfp4-group-size", type=int, default=None)
    parser.add_argument("--nvfp4-scale-rule", choices=["static_6", "four_over_six_mse"], default="four_over_six_mse")
    parser.add_argument("--cache-dir", default="/home/agent/wja/.cache/huggingface")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Llama compression")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    model_root = model_result_root(args.output_root, args.model)
    out_dir = model_root / "prepared" / args.method
    model_out = out_dir / "model.pt"
    metadata_out = out_dir / "metadata.json"
    if args.skip_existing and model_out.exists() and metadata_out.exists():
        print(f"skip existing: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "compression_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    calib = CalibConfig(
        samples=args.calib_samples,
        seq_len=args.seq_len,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    blocks, calib_metadata = build_wikitext2_blocks(calib, model_key=args.model)
    write_json(model_root / "calibration" / "wikitext2_metadata.json", calib_metadata)
    loader = build_calib_loader(blocks, batch_size=args.calib_batch_size)

    config = CompressionConfig(
        method=args.method,
        sparsity=args.sparsity,
        sparsegpt_block_size=args.sparsegpt_block_size,
        sparsegpt_percdamp=args.sparsegpt_percdamp,
        nvfp4_group_size=args.nvfp4_group_size or default_group_size(args.method),
        nvfp4_scale_rule=args.nvfp4_scale_rule,
    )

    spec = model_spec(args.model)
    print(f"loading model={spec['path']} method={args.method} gpu={args.gpu}")
    model = load_model(args.model, device=device, dtype=dtype)
    modules = compressible_modules(model, args.model)
    print(f"compressible modules: {len(modules)}")
    module_records: list[dict[str, Any]] = []

    if args.method in {"dense_nvfp4", "marlin_nvfp4"}:
        hdiag = collect_hessian_diag_llama(model, modules, loader, device=device)
        for index, info in enumerate(modules, start=1):
            row = compress_nvfp4_module(info, config, hdiag.get(info.name))
            row.update({"index": index, "name": info.name})
            append_jsonl(log_path, row)
            module_records.append(row)
    elif args.method in {"sparse_bf16", "sparse_nvfp4"}:
        index = 0
        for layer_name, layer_modules in grouped_by_layer(modules):
            print(f"sparse layer {layer_name}: modules={len(layer_modules)}")
            hessians = collect_hessians_for_modules_llama(model, layer_modules, loader, device=device)
            for info in layer_modules:
                index += 1
                hessian, hessian_samples = hessians[info.name]
                row = compress_sparse_module(info, hessian, hessian_samples, config=config)
                row.update({"index": index, "name": info.name, "layer": layer_name})
                append_jsonl(log_path, row)
                module_records.append(row)
            del hessians
            cleanup_cuda()
    else:
        raise ValueError(args.method)

    payload = {
        "state_dict": cpu_state_dict(model),
        "metadata": {
            "checkpoint_format": "llama2_real_compressed_dense_state_v1",
            "model_key": args.model,
            "model_label": spec["label"],
            "model_path": spec["path"],
            "model_family": spec["family"],
            "method": args.method,
            "dtype": args.dtype,
            "compression_config": config.__dict__,
            "calibration": calib_metadata,
            "selected_modules": len(modules),
            "compressed_modules": sum(1 for row in module_records if row.get("status") == "ok"),
            "skipped": [row for row in module_records if row.get("status") != "ok"],
            "timestamp": utc_now(),
        },
    }
    torch.save(payload, model_out)
    write_json(metadata_out, payload["metadata"])
    print(f"prepared {args.method}: {model_out}")


def local_cuda_index(requested_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA is required for Llama compression")
    if requested_gpu < count:
        return requested_gpu
    if visible:
        return 0
    return requested_gpu


def default_group_size(method: str) -> int:
    if method == "sparse_nvfp4":
        return 32
    return 16


def grouped_by_layer(modules: list[Any]) -> list[tuple[str, list[Any]]]:
    groups: dict[str, list[Any]] = {}
    order: list[str] = []
    for info in modules:
        parts = info.name.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            layer = ".".join(parts[:3])
        elif len(parts) >= 3 and parts[0] == "language_model" and parts[1] == "layers":
            layer = ".".join(parts[:3])
        else:
            layer = "__other__"
        if layer not in groups:
            groups[layer] = []
            order.append(layer)
        groups[layer].append(info)
    return [(layer, groups[layer]) for layer in order]


def compress_nvfp4_module(info: Any, config: CompressionConfig, hdiag: torch.Tensor | None) -> dict[str, Any]:
    _parent, _child_name = module_parent_from_info(info)
    weight = info.module.weight.data
    qweight, scales, qstats = nvfp4_quantize_weight(weight, config, hdiag)
    if qstats.get("status") != "ok":
        return {"status": "skipped", "reason": qstats.get("reason", "quant_failed"), "quant": qstats}
    info.module.weight.data.copy_(qweight.reshape_as(info.module.weight.data))
    return {
        "status": "ok",
        "algorithm": "activation_calibrated_nvfp4_ptq",
        "hessian_diag_mean": float(hdiag.float().mean().item()) if hdiag is not None else "",
        "hessian_diag_max": float(hdiag.float().max().item()) if hdiag is not None else "",
        "quant": qstats,
        "scale_shape": list(scales.shape) if scales is not None else None,
    }


def compress_sparse_module(
    info: Any,
    hessian: torch.Tensor,
    hessian_samples: int,
    *,
    config: CompressionConfig,
) -> dict[str, Any]:
    pattern = "dense_2_4" if config.method == "sparse_bf16" else "nvfp4_pair_2_4_over_8"
    try:
        pruned, mask, prune_stats = sparsegpt_prune_weight(
            info.module.weight.data,
            hessian,
            pattern=pattern,
            sparsity=config.sparsity,
            block_size=config.sparsegpt_block_size,
            percdamp=config.sparsegpt_percdamp,
            module_name=info.name,
        )
        info.module.weight.data.copy_(pruned.reshape_as(info.module.weight.data))
        row: dict[str, Any] = {
            "status": "ok",
            "hessian_samples": hessian_samples,
            "prune": prune_stats,
            "mask_shape": list(mask.shape),
        }
        if config.method == "sparse_nvfp4":
            qweight, scales, qstats = nvfp4_quantize_weight(info.module.weight.data, config, torch.diag(hessian).detach().cpu())
            if qstats.get("status") != "ok":
                row.update({"status": "skipped", "reason": qstats.get("reason", "quant_failed"), "quant": qstats})
            else:
                info.module.weight.data.copy_(qweight.reshape_as(info.module.weight.data))
                row["quant"] = qstats
                row["scale_shape"] = list(scales.shape) if scales is not None else None
        return row
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}:{exc}"}


def module_parent_from_info(info: Any) -> tuple[Any, str]:
    return None, info.name.rsplit(".", 1)[-1]


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda()
