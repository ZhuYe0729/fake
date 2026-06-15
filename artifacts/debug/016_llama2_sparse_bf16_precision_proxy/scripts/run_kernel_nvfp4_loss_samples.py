#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import torch

from common_sparse_bf16_proxy import DEBUG_ROOT, FAKE_ROOT, SOURCE_ROOT, policy_paths, read_csv, selected_from_text, write_csv, write_json

SOURCE_015_ROOT = FAKE_ROOT / "artifacts/debug/015_llama2_prefill_kernel_loss_modeling"
SOURCE_015_SCRIPTS = SOURCE_015_ROOT / "scripts"
if str(SOURCE_015_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_015_SCRIPTS))

from common_kernel_prefill_loss import (  # type: ignore  # noqa: E402
    METHODS,
    cleanup_cuda,
    compressible_modules,
    compute_nll,
    dtype_from_arg,
    install_kernel_modules,
    load_calibration_blocks,
    load_llama_for_quality,
    load_prepared_payload,
    local_cuda_index,
    parse_methods,
    quality_config_from_args,
    restore_original_modules,
    save_original_modules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sampled dense/sparse NVFP4 prefill loss policies through real kernels.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--max-policies", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--policies-csv", type=Path, default=None)
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    methods = parse_methods(args.methods)
    paths = policy_paths(args.output_root)
    policies_csv = args.policies_csv or paths["policies"]
    if not policies_csv.exists():
        raise FileNotFoundError(f"Missing policies: {policies_csv}")
    policies = read_csv(policies_csv)
    if args.max_policies is not None:
        policies = policies[: args.max_policies]
    gpus = resolve_gpus(args.gpus)
    if not gpus:
        raise ValueError("--gpus must list at least one GPU")

    for method in methods:
        completed = completed_policy_ids(args.output_root, method, args.output_tag) if args.skip_existing else set()
        pending = [row for row in policies if row["policy_id"] not in completed]
        worker_args = []
        for worker_index, gpu in enumerate(gpus):
            worker_policies = [row for idx, row in enumerate(pending) if idx % len(gpus) == worker_index]
            worker_args.append((args, method, worker_index, gpu, worker_policies))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(gpus)) as pool:
            pool.map(run_worker, worker_args)
        merge_worker_outputs(args.output_root, method, args.output_tag)
        write_json(
            loss_path(args.output_root, method, args.output_tag).with_suffix(".metadata.json"),
            {
                "method": method,
                "source_root": str(args.source_root),
                "policies_csv": str(policies_csv),
                "output_tag": args.output_tag,
                "dtype": args.dtype,
                "gpus": gpus,
                "calib_samples": args.calib_samples,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "requested_policies": len(policies),
                "pending_policies": len(pending),
                "validity": "kernel_aware_real_runtime_forward_with_activation_quantization",
            },
        )
        print(f"wrote merged loss rows to {loss_path(args.output_root, method, args.output_tag)}")


def run_worker(payload: tuple[argparse.Namespace, str, int, int, list[dict[str, Any]]]) -> None:
    args, method, worker_index, gpu, policies = payload
    if not policies:
        return
    local_gpu = local_cuda_index(gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    config = quality_config_from_args(args)
    blocks, _ = load_calibration_blocks(config)
    model = load_llama_for_quality(device=device, dtype=dtype)
    modules = compressible_modules(model, "llama2-7b")
    module_names = {info.name for info in modules}
    state, source_metadata = load_prepared_payload(args.source_root, method)
    dense = compute_nll(model, blocks, device=device, batch_size=args.batch_size)
    output_path = worker_output_path(args.output_root, method, args.output_tag, worker_index)
    rows = read_csv(output_path) if args.skip_existing and output_path.exists() and output_path.stat().st_size else []
    done = {row["policy_id"] for row in rows}

    for policy in policies:
        policy_id = policy["policy_id"]
        if policy_id in done:
            continue
        selected = selected_from_text(policy["selected_names"])
        missing = sorted(selected - module_names)
        if missing:
            raise KeyError(f"policy {policy_id} references unknown modules: {missing[:5]}")
        print(f"worker={worker_index} gpu={gpu} method={method} policy={policy_id} selected={len(selected)}")
        saved = save_original_modules(model, selected)
        runtime_report: dict[str, Any] = {}
        try:
            runtime_report = install_kernel_modules(model, modules, method=method, state=state, selected_names=selected, dtype=dtype)
            loss = compute_nll(model, blocks, device=device, batch_size=args.batch_size)
        finally:
            restore_original_modules(model, saved)
            cleanup_cuda()
        rows.append(
            {
                "policy_id": policy_id,
                "sample_kind": policy["sample_kind"],
                "method": method,
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
                "worker_index": worker_index,
                "gpu": gpu,
                "source_artifact": source_metadata.get("artifact", ""),
            }
        )
        write_csv(output_path, rows)

    del model
    gc.collect()
    cleanup_cuda()


def resolve_gpus(spec: str | None) -> list[int]:
    if spec:
        return [int(item) for item in spec.split(",") if item.strip()]
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) >= 4:
        return [0, 1, 2, 3]
    return [1, 2, 3, 4]


def loss_path(output_root: Path, method: str, output_tag: str) -> Path:
    suffix = f"_{output_tag}" if output_tag else ""
    return output_root / "loss" / f"loss_samples_{method}{suffix}.csv"


def worker_output_path(output_root: Path, method: str, output_tag: str, worker_index: int) -> Path:
    suffix = f"_{output_tag}" if output_tag else ""
    return output_root / "loss" / "kernel_workers" / f"loss_samples_{method}{suffix}_worker_{worker_index:02d}.csv"


def completed_policy_ids(output_root: Path, method: str, output_tag: str) -> set[str]:
    rows: list[dict[str, Any]] = []
    suffix = f"_{output_tag}" if output_tag else ""
    paths = [loss_path(output_root, method, output_tag), *sorted((output_root / "loss" / "kernel_workers").glob(f"loss_samples_{method}{suffix}_worker_*.csv"))]
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            rows.extend(read_csv(path))
    return {row["policy_id"] for row in rows}


def merge_worker_outputs(output_root: Path, method: str, output_tag: str) -> None:
    rows: list[dict[str, Any]] = []
    suffix = f"_{output_tag}" if output_tag else ""
    for path in sorted((output_root / "loss" / "kernel_workers").glob(f"loss_samples_{method}{suffix}_worker_*.csv")):
        if path.exists() and path.stat().st_size > 0:
            rows.extend(read_csv(path))
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[row["policy_id"]] = row
    merged = sorted(dedup.values(), key=lambda row: row["policy_id"])
    write_csv(loss_path(output_root, method, output_tag), merged)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cuda()
