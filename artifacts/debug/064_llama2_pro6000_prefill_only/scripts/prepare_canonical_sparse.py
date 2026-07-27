#!/usr/bin/env python3
"""Prepare real calibrated uniform compressed Llama-2-7B-Chat artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent / "runs/experiment/canonical"

import compression_common as compression  # noqa: E402
from compression_common import (  # noqa: E402
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
    model_spec,
    nvfp4_quantize_weight,
    sparsegpt_prune_weight,
    utc_now,
    write_json,
)


MODEL_KEY = "llama2-7b-chat"
MODEL_PATH = Path(os.environ.get("COSPAQ_MODEL_PATH", "/root/data/models/Llama-2-7b-chat-hf"))
METHODS = ("sparse_bf16", "sparse_nvfp4")

compression.MODELS[MODEL_KEY] = {
    "label": "Llama-2-7B-Chat",
    "path": str(MODEL_PATH),
    "family": "llama",
    "loader": "causal_lm",
    "trust_remote_code": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("COSPAQ_MODEL_PATH", MODEL_PATH)))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--sparsegpt-block-size", type=int, default=128)
    parser.add_argument("--sparsegpt-percdamp", type=float, default=0.01)
    parser.add_argument("--nvfp4-group-size", type=int, default=None)
    parser.add_argument(
        "--nvfp4-scale-rule",
        choices=["static_6", "four_over_six_mse"],
        default="four_over_six_mse",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE", "/root/data/huggingface/datasets"))
    parser.add_argument("--dataset-arrow", type=Path, default=Path(os.environ.get("COSPAQ_WIKITEXT_ARROW", "")))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--sparse-nvfp4-prequant-only",
        action="store_true",
        help=("For sparse_nvfp4, retain the calibrated SparseGPT BF16 weights "
              "and leave the single final NVFP4 conversion to the phase exporter."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression.MODELS[MODEL_KEY]["path"] = str(args.model_path)
    methods = parse_methods(args.methods)
    model_path = args.model_path.resolve()
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"missing model config: {model_path / 'config.json'}")
    compression.MODELS[MODEL_KEY]["path"] = str(model_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Llama compression")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)

    for method in methods:
        prepare_one_method(args, method, device=f"cuda:{local_gpu}")


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={list(METHODS)}")
    return methods


def local_cuda_index(requested_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    if requested_gpu < count:
        return requested_gpu
    if visible:
        return 0
    return requested_gpu


def prepare_one_method(args: argparse.Namespace, method: str, *, device: str) -> None:
    if args.sparse_nvfp4_prequant_only and method != "sparse_nvfp4":
        raise ValueError("--sparse-nvfp4-prequant-only requires --methods sparse_nvfp4")
    out_dir = args.output_root / "prepared" / method
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
    blocks, calib_metadata = build_wikitext2_blocks(
        calib, model_key=MODEL_KEY, tokenizer_path=args.model_path.resolve(),
        dataset_arrow_path=args.dataset_arrow
    )
    write_json(args.output_root / "calibration" / f"wikitext2_{method}_metadata.json", calib_metadata)
    loader = build_calib_loader(blocks, batch_size=args.calib_batch_size)
    config = CompressionConfig(
        method=method,
        sparsity=args.sparsity,
        sparsegpt_block_size=args.sparsegpt_block_size,
        sparsegpt_percdamp=args.sparsegpt_percdamp,
        nvfp4_group_size=args.nvfp4_group_size or default_group_size(method),
        nvfp4_scale_rule=args.nvfp4_scale_rule,
    )

    dtype = dtype_from_arg(args.dtype)
    spec = model_spec(MODEL_KEY)
    print(f"loading model={spec['path']} method={method} device={device}", flush=True)
    model = load_model(MODEL_KEY, device=device, dtype=dtype)
    modules = compressible_modules(model, MODEL_KEY)
    module_records: list[dict[str, Any]] = []
    print(f"compressible modules: {len(modules)}", flush=True)

    if method in {"dense_nvfp4", "marlin_nvfp4"}:
        hdiag = collect_hessian_diag_llama(model, modules, loader, device=device)
        for index, info in enumerate(modules, start=1):
            row = compress_nvfp4_module(info, config, hdiag.get(info.name))
            row.update({"index": index, "name": info.name})
            append_jsonl(log_path, row)
            module_records.append(row)
    elif method in {"sparse_bf16", "sparse_nvfp4"}:
        index = 0
        for layer_name, layer_modules in grouped_by_layer(modules):
            print(f"sparse layer {layer_name}: modules={len(layer_modules)}", flush=True)
            hessians = collect_hessians_for_modules_llama(
                model, layer_modules, loader, device=device
            )
            for info in layer_modules:
                index += 1
                hessian, hessian_samples = hessians[info.name]
                row = compress_sparse_module(
                    info,
                    hessian,
                    hessian_samples,
                    config=config,
                    skip_nvfp4_quantization=args.sparse_nvfp4_prequant_only,
                )
                row.update({"index": index, "name": info.name, "layer": layer_name})
                append_jsonl(log_path, row)
                module_records.append(row)
            del hessians
            cleanup_cuda()
    else:
        raise ValueError(method)

    payload = {
        "state_dict": cpu_state_dict(model),
        "metadata": {
            "checkpoint_format": "llama2_chat_real_compressed_dense_state_v1",
            "model_key": MODEL_KEY,
            "model_label": spec["label"],
            "model_path": spec["path"],
            "model_family": spec["family"],
            "method": method,
            "sparse_nvfp4_prequant_only": args.sparse_nvfp4_prequant_only,
            "dtype": args.dtype,
            "compression_config": config.__dict__,
            "calibration": calib_metadata,
            "selected_modules": len(modules),
            "compressed_modules": sum(
                1 for row in module_records if row.get("status") == "ok"
            ),
            "skipped": [row for row in module_records if row.get("status") != "ok"],
            "timestamp": utc_now(),
        },
    }
    torch.save(payload, model_out)
    write_json(metadata_out, payload["metadata"])
    print(f"prepared {method}: {model_out}", flush=True)

    del model
    cleanup_cuda()


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
        else:
            layer = "__other__"
        if layer not in groups:
            groups[layer] = []
            order.append(layer)
        groups[layer].append(info)
    return [(layer, groups[layer]) for layer in order]


def compress_nvfp4_module(
    info: Any, config: CompressionConfig, hdiag: torch.Tensor | None
) -> dict[str, Any]:
    weight = info.module.weight.data
    qweight, scales, qstats = nvfp4_quantize_weight(weight, config, hdiag)
    if qstats.get("status") != "ok":
        return {
            "status": "skipped",
            "reason": qstats.get("reason", "quant_failed"),
            "quant": qstats,
        }
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
    skip_nvfp4_quantization: bool = False,
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
        if config.method == "sparse_nvfp4" and not skip_nvfp4_quantization:
            qweight, scales, qstats = nvfp4_quantize_weight(
                info.module.weight.data, config, torch.diag(hessian).detach().cpu()
            )
            if qstats.get("status") != "ok":
                row.update(
                    {
                        "status": "skipped",
                        "reason": qstats.get("reason", "quant_failed"),
                        "quant": qstats,
                    }
                )
            else:
                info.module.weight.data.copy_(qweight.reshape_as(info.module.weight.data))
                row["quant"] = qstats
                row["scale_shape"] = list(scales.shape) if scales is not None else None
        elif config.method == "sparse_nvfp4":
            row["quant"] = {"status": "deferred_to_phase_exporter"}
        return row
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}:{exc}"}


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda()
