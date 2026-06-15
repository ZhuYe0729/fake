#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from common_dialogsum import (
    DEBUG_ROOT,
    DEFAULT_MODEL_KEY,
    SOURCE_ROOT,
    UNIFORM_METHODS,
    cleanup_cuda,
    dtype_from_arg,
    install_uniform_runtime,
    load_compressed_state_into_model,
    load_eval_model,
    local_cuda_index,
    replacement_report_dict,
    write_csv,
    write_json,
)

SCENARIO = {
    "name": "normal_02",
    "batch_size": 1,
    "input_tokens": 16384,
    "output_tokens": 256,
    "m_prefill": 16384,
    "m_decode": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate uniform methods with real normal02 E2E latency.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--methods", default=",".join(UNIFORM_METHODS))
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = dtype_from_arg(args.dtype)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in UNIFORM_METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    rows = []
    for method in methods:
        print(f"e2e validating uniform method={method}")
        try:
            model = load_eval_model(dtype=dtype, device=device, attn=args.attn)
            compression_meta = load_compressed_state_into_model(
                model,
                method=method,
                source_root=args.source_root,
                model_key=DEFAULT_MODEL_KEY,
                device=device,
            )
            report = install_uniform_runtime(model, method=method, model_key=DEFAULT_MODEL_KEY, dtype=dtype)
            result = benchmark(model, device=device, warmup_iters=args.warmup_iters, iters=args.iters)
            row = {
                "method": method,
                "e2e_status": "ok",
                "unsupported_reason": "",
                "e2e_total_mean_ms": result["total_mean"],
                "e2e_total_median_ms": result["total_median"],
                "e2e_total_min_ms": result["total_min"],
                "e2e_total_max_ms": result["total_max"],
                "e2e_prefill_mean_ms": result["prefill_mean"],
                "e2e_decode_avg_mean_ms": result["decode_avg_mean"],
                "e2e_decode_first_mean_ms": result["decode_first_mean"],
                "e2e_decode_steady_mean_ms": result["decode_steady_mean"],
                "e2e_times_ms": ";".join(f"{t:.6f}" for t in result["total_times"]),
                "runtime_report": replacement_report_dict(report) if report is not None else None,
                "compression_artifact": "" if compression_meta is None else compression_meta.get("artifact", ""),
                "requested_gpu": args.gpu,
                "local_gpu": local_gpu,
                "iters": args.iters,
                "warmup_iters": args.warmup_iters,
            }
            rows.append(row)
            del model
        except Exception as exc:
            rows.append(
                {
                    "method": method,
                    "e2e_status": "error",
                    "unsupported_reason": f"{type(exc).__name__}: {exc}",
                    "requested_gpu": args.gpu,
                    "local_gpu": local_gpu,
                    "iters": args.iters,
                    "warmup_iters": args.warmup_iters,
                }
            )
        gc.collect()
        cleanup_cuda()
    out_csv = args.output_csv or args.output_root / "speed" / "uniform_e2e_validation.csv"
    write_csv(out_csv, rows)
    write_json(
        out_csv.with_suffix(".metadata.json"),
        {
            "scenario": SCENARIO,
            "methods": methods,
            "gpu": args.gpu,
            "dtype": args.dtype,
            "attn": args.attn,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
        },
    )
    print(f"wrote {len(rows)} rows to {out_csv}")


@torch.inference_mode()
def benchmark(model, *, device: str, warmup_iters: int, iters: int) -> dict[str, Any]:
    batch_size = SCENARIO["batch_size"]
    input_tokens = SCENARIO["input_tokens"]
    output_tokens = SCENARIO["output_tokens"]
    for _ in range(warmup_iters):
        ids = torch.randint(0, 1000, (1, 16), device=device)
        out = model(ids, use_cache=False)
        del ids, out
    torch.cuda.synchronize()
    gc.collect()
    cleanup_cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_times: list[float] = []
    prefill_times: list[float] = []
    decode_avg_times: list[float] = []
    decode_first_times: list[float] = []
    decode_steady_times: list[float] = []
    for _ in range(iters):
        input_ids = torch.randint(0, 1000, (batch_size, input_tokens), device=device)
        start.record()
        out = model(input_ids, use_cache=(output_tokens > 0))
        end.record()
        torch.cuda.synchronize()
        prefill_ms = float(start.elapsed_time(end))
        prefill_times.append(prefill_ms)
        past_key_values = out.past_key_values
        next_token = torch.randint(0, 1000, (batch_size, 1), device=device)
        decode_times = []
        for _ in range(output_tokens):
            start.record()
            out = model(next_token, past_key_values=past_key_values, use_cache=True)
            end.record()
            torch.cuda.synchronize()
            decode_times.append(float(start.elapsed_time(end)))
            past_key_values = out.past_key_values
            next_token = torch.randint(0, 1000, (batch_size, 1), device=device)
        total_times.append(prefill_ms + sum(decode_times))
        decode_avg_times.append(sum(decode_times) / len(decode_times))
        decode_first_times.append(decode_times[0])
        decode_steady_times.append(sum(decode_times[2:]) / max(len(decode_times[2:]), 1))
        del input_ids, out, past_key_values, next_token, decode_times
        gc.collect()
        cleanup_cuda()
    return {
        "total_times": total_times,
        "total_mean": mean(total_times),
        "total_median": median(total_times),
        "total_min": min(total_times),
        "total_max": max(total_times),
        "prefill_mean": mean(prefill_times),
        "decode_avg_mean": mean(decode_avg_times),
        "decode_first_mean": mean(decode_first_times),
        "decode_steady_mean": mean(decode_steady_times),
    }


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        cleanup_cuda()
