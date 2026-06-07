#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import lm_eval
import torch

from common import (
    DEFAULT_MODEL_KEY,
    EXPERIMENT_ROOT,
    METHODS,
    cleanup_cuda,
    compressible_modules,
    dtype_from_arg,
    load_model,
    model_result_root,
    model_spec,
    replacement_report_dict,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prepared Llama-2-7B compressed artifact on arc_easy.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--tasks", default="arc_easy")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Llama evaluation")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    spec = model_spec(args.model)
    model_root = model_result_root(args.output_root, args.model)
    out_dir = model_root / "methods" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, device=device, dtype=dtype)
    compression_metadata: dict[str, Any] | None = None
    report = None
    if args.method != "dense_bf16":
        source_method = "dense_nvfp4" if args.method == "dense_nvfp4_prefill_marlin_decode" else args.method
        artifact = model_root / "prepared" / source_method / "model.pt"
        payload = torch.load(artifact, map_location="cpu")
        compression_metadata = dict(payload.get("metadata", {}))
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Failed to load compressed state: missing={missing} unexpected={unexpected}")
        model.to(device)
        report = install_runtime_kernel(model, args.method, model_key=args.model, dtype=dtype)

    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=spec["path"],
        backend="causal",
        dtype=dtype,
        device=device,
        batch_size=args.batch_size,
        trust_remote_code=bool(spec["trust_remote_code"]),
    )
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate returned None")
    payload = {
        "method": args.method,
        "model_key": args.model,
        "model_label": spec["label"],
        "model_path": spec["path"],
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "timestamp": utc_now(),
        "results": results["results"],
    }
    write_json(out_dir / "accuracy.json", payload)
    write_json(
        out_dir / "eval_metadata.json",
        {
            "method": args.method,
            "gpu": args.gpu,
            "dtype": args.dtype,
            "compression_metadata": compression_metadata,
            "runtime_report": replacement_report_dict(report) if report is not None else None,
            "timestamp": utc_now(),
        },
    )
    print(json.dumps(payload["results"], indent=2, default=str))


def local_cuda_index(requested_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA is required for Llama evaluation")
    if requested_gpu < count:
        return requested_gpu
    if visible:
        return 0
    return requested_gpu


def install_runtime_kernel(model: torch.nn.Module, method: str, *, model_key: str, dtype: torch.dtype):
    family = model_spec(model_key)["family"]
    if method == "dense_nvfp4":
        from fake.kernels.cutlass_nvfp4 import CutlassNVFP4Config, replace_linear_with_cutlass_nvfp4

        return replace_linear_with_cutlass_nvfp4(model, family, CutlassNVFP4Config())
    if method == "sparse_bf16":
        from fake.kernels.cutlass_sparse_bf16 import CutlassSparseBF16Config, replace_linear_with_cutlass_sparse_bf16

        return replace_linear_with_cutlass_sparse_bf16(model, family, CutlassSparseBF16Config(prune=False))
    if method == "sparse_nvfp4":
        from fake.kernels.cutlass_sparse_nvfp4 import CutlassSparseNVFP4Config, replace_linear_with_cutlass_sparse_nvfp4

        return replace_linear_with_cutlass_sparse_nvfp4(model, family, CutlassSparseNVFP4Config(prune=False))
    if method == "marlin_nvfp4":
        from fake.kernels.marlin_nvfp4 import MarlinNVFP4Config, replace_linear_with_marlin_nvfp4

        return replace_linear_with_marlin_nvfp4(model, family, MarlinNVFP4Config(activation_dtype=dtype))
    if method == "dense_nvfp4_prefill_marlin_decode":
        from fake.models.qwen3_5_kernels import QwenHybridDenseNVFP4Linear
        from fake.models.qwen3_5_kernels import _load_wrapper

        wrapper = _load_wrapper()
        skipped = []
        replaced = 0
        for info in compressible_modules(model, model_key):
            if info.kind != "linear":
                continue
            parent = model
            parts = info.name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            linear = getattr(parent, parts[-1])
            try:
                canonical = wrapper.canonical_from_linear(linear, device=linear.weight.device)
                setattr(
                    parent,
                    parts[-1],
                    QwenHybridDenseNVFP4Linear(
                        canonical,
                        decode_activation_dtype=dtype,
                        marlin_m_threshold=1,
                        prefill_backend="dense_nvfp4",
                        decode_backend="marlin_nvfp4",
                    ),
                )
                replaced += 1
            except Exception as exc:
                skipped.append({"name": info.name, "reason": f"{type(exc).__name__}:{exc}"})
        return {
            "backend": f"{model_key}_dense_nvfp4_prefill_marlin_decode",
            "config": {"source_artifact": "dense_nvfp4", "marlin_m_threshold": 1},
            "replaced_linear_count": replaced,
            "skipped_linear_count": len(skipped),
            "skipped": skipped,
        }
    raise ValueError(method)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda()
