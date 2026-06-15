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
    cleanup_cuda,
    compressible_modules,
    compute_nll,
    dtype_from_arg,
    install_kernel_modules,
    load_calibration_blocks,
    load_llama_for_quality,
    load_prepared_payload,
    local_cuda_index,
    loss_ablation_metadata,
    parse_loss_policies,
    parse_methods,
    quality_config_from_args,
    restore_original_modules,
    save_original_modules,
    select_modules,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prefill-only WikiText-2 loss ablations through real NVFP4 kernels.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--policies", default="default")
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--max-policies", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
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
    output_csv = args.output_csv or args.output_root / "ablations" / f"kernel_loss_ablation_{'_'.join(methods)}.csv"
    rows = read_existing(output_csv) if args.skip_existing else []
    done = {(row["method"], row["policy"]) for row in rows}

    model = load_llama_for_quality(device=device, dtype=dtype)
    modules = select_modules(compressible_modules(model, "llama2-7b"), args.max_modules)
    module_names = [info.name for info in modules]
    policies = parse_loss_policies(args.policies, module_names)
    if args.max_policies is not None:
        policies = policies[: args.max_policies]
    states: dict[str, dict[str, torch.Tensor]] = {}
    source_metadata: dict[str, Any] = {}
    for method in methods:
        states[method], source_metadata[method] = load_prepared_payload(args.source_root, method)

    dense = compute_nll(model, blocks, device=device, batch_size=args.batch_size)
    if ("dense_bf16", "none") not in done:
        rows.append(make_dense_row(dense))
        write_csv(output_csv, rows)

    for method in methods:
        state = states[method]
        for policy in policies:
            key = (method, policy["policy"])
            if key in done:
                print(f"skipping existing method={method} policy={policy['policy']}")
                continue
            selected = set(policy["selected"])
            if not selected:
                continue
            print(f"running kernel method={method} policy={policy['policy']} selected={len(selected)}")
            saved = save_original_modules(model, selected)
            runtime_report: dict[str, Any] = {}
            try:
                runtime_report = install_kernel_modules(
                    model,
                    modules,
                    method=method,
                    state=state,
                    selected_names=selected,
                    dtype=dtype,
                )
                loss = compute_nll(model, blocks, device=device, batch_size=args.batch_size)
            finally:
                restore_original_modules(model, saved)
                cleanup_cuda()
            rows.append(
                {
                    "method": method,
                    "policy": policy["policy"],
                    "policy_kind": policy["policy_kind"],
                    "layer": policy["layer"],
                    "linear_type": policy["linear_type"],
                    "selected_modules": len(selected),
                    "replaced_modules": runtime_report.get("replaced_modules", 0),
                    "runtime_path": runtime_report.get("runtime_path", ""),
                    "activation_quantization": runtime_report.get("activation_quantization", ""),
                    "loss": loss["nll"],
                    "loss_delta_vs_dense": loss["nll"] - dense["nll"],
                    "ppl": loss["ppl"],
                    "loss_tokens": loss["tokens"],
                    "loss_sum": loss["loss_sum"],
                    "dense_loss": dense["nll"],
                }
            )
            write_csv(output_csv, rows)
    write_json(
        output_csv.with_suffix(".metadata.json"),
        loss_ablation_metadata(args, calib_metadata, methods, policies, source_metadata),
    )
    print(f"wrote {len(rows)} rows to {output_csv}")


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    import csv

    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def make_dense_row(dense: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "dense_bf16",
        "policy": "none",
        "policy_kind": "baseline",
        "layer": "",
        "linear_type": "",
        "selected_modules": 0,
        "replaced_modules": 0,
        "runtime_path": "dense_bf16",
        "activation_quantization": "none",
        "loss": dense["nll"],
        "loss_delta_vs_dense": 0.0,
        "ppl": dense["ppl"],
        "loss_tokens": dense["tokens"],
        "loss_sum": dense["loss_sum"],
        "dense_loss": dense["nll"],
    }


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        cleanup_cuda()
