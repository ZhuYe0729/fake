#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lm_eval
import torch

from common_quality import (
    COMPRESSED_CORE_METHODS,
    DEBUG_ROOT,
    MODEL_KEY,
    QualityConfig,
    apply_compressed_weights,
    cleanup_cuda,
    compressible_modules,
    compute_nll,
    default_ablation_policies,
    dtype_from_arg,
    load_calibration_blocks,
    load_llama_for_quality,
    local_cuda_index,
    model_spec,
    parse_policy,
    write_csv,
    write_json,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Llama2 mixed-policy quality ablations.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEBUG_ROOT.parents[1] / "results/main/003_llama2_7b_arc_easy_accuracy")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default=",".join(COMPRESSED_CORE_METHODS))
    parser.add_argument("--policies", default="default")
    parser.add_argument("--task", default="arc_easy")
    parser.add_argument("--arc-limit", type=int, default=128)
    parser.add_argument("--full-arc", action="store_true")
    parser.add_argument("--skip-arc", action="store_true")
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
    policies = default_ablation_policies() if args.policies == "default" else [p.strip() for p in args.policies.split(",") if p.strip()]
    arc_limit = None if args.full_arc else args.arc_limit

    config = QualityConfig(
        calib_samples=args.calib_samples,
        seq_len=args.seq_len,
        seed=args.seed,
        batch_size=args.batch_size,
        source_root=args.source_root,
        output_root=args.output_root,
    )
    blocks, calib_metadata = load_calibration_blocks(config)
    rows: list[dict[str, Any]] = []
    dense_nll: dict[str, Any] | None = None
    spec = model_spec(MODEL_KEY)

    for method in ["dense_bf16", *methods]:
        method_policies = ["none"] if method == "dense_bf16" else policies
        for policy in method_policies:
            print(f"running method={method} policy={policy}")
            model = load_llama_for_quality(device=device, dtype=dtype)
            modules = compressible_modules(model, MODEL_KEY)
            module_names = [info.name for info in modules]
            selected = parse_policy(policy, module_names)
            replaced = apply_compressed_weights(
                model,
                modules,
                source_root=args.source_root,
                method=method,
                selected_names=selected,
            )
            nll = compute_nll(model, blocks, device=device, batch_size=args.batch_size)
            if method == "dense_bf16":
                dense_nll = nll
            row: dict[str, Any] = {
                "method": method,
                "policy": policy,
                "selected_modules": len(selected),
                "replaced_modules": replaced,
                "nll": nll["nll"],
                "ppl": nll["ppl"],
                "tokens": nll["tokens"],
                "nll_delta_vs_dense": nll["nll"] - dense_nll["nll"] if dense_nll is not None else 0.0,
            }
            if not args.skip_arc:
                arc = run_arc_eval(model, spec, task=args.task, dtype=dtype, device=device, batch_size="auto", limit=arc_limit)
                row.update(
                    {
                        "task": args.task,
                        "arc_limit": arc_limit,
                        "arc_acc": arc.get("acc,none"),
                        "arc_acc_norm": arc.get("acc_norm,none"),
                        "arc_sample_len": arc.get("sample_len"),
                    }
                )
            rows.append(row)
            out_dir = args.output_root / "ablations" / method / sanitize(policy)
            write_json(out_dir / "result.json", row)
            del model
            cleanup_cuda()

    write_csv(args.output_root / "ablations" / "policy_quality_results.csv", rows)
    write_run_metadata(
        args.output_root / "ablations" / "run_ablation_metadata.json",
        {
            "source_root": str(args.source_root),
            "dtype": args.dtype,
            "gpu": args.gpu,
            "methods": methods,
            "policies": policies,
            "task": args.task,
            "arc_limit": arc_limit,
            "skip_arc": args.skip_arc,
            "calibration": calib_metadata,
        },
    )
    print(f"wrote {len(rows)} ablation rows")


def sanitize(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def run_arc_eval(
    model,
    spec: dict[str, Any],
    *,
    task: str,
    dtype: torch.dtype,
    device: str,
    batch_size: str,
    limit: int | None,
) -> dict[str, Any]:
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=spec["path"],
        backend="causal",
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        trust_remote_code=bool(spec["trust_remote_code"]),
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate returned None")
    return dict(results["results"][task])


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda()
