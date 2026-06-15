#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from common_pareto import DEBUG_ROOT, SCENARIO, f, read_csv, read_json, write_csv, write_json
from real_policy_runtime import apply_real_policy_runtime

FAKE_ROOT = Path(__file__).resolve().parents[4]
if str(FAKE_ROOT) not in sys.path:
    sys.path.insert(0, str(FAKE_ROOT))

MODEL_PATH = "/home/agent/wja/data/models/LLM-Research/llama-2-7b"
PREPARED_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selected Pareto policies with real full-model prefill-only E2E latency.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--points", default="validation", help="'validation' or comma-separated point indices.")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--point-output-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    point_indices = select_point_indices(args)
    rows = []
    for point_index in point_indices:
        policy_path = find_pareto_policy(args.output_root, point_index)
        print(f"e2e validating point={point_index} policy={policy_path}")
        model = load_model(dtype=dtype, attn=args.attn, device=device)
        policy = read_json(policy_path)
        report = apply_real_policy_runtime(model, policy, prepared_root=PREPARED_ROOT, activation_dtype=dtype)
        result = benchmark_prefill(model, device=device, warmup_iters=args.warmup_iters, iters=args.iters)
        summary = read_json(policy_path)["summary"]
        row = {
            "point_index": point_index,
            "policy_json": str(policy_path),
            "converted_policy_json": "",
            "predicted_linear_latency_ms": summary["latency_ms"],
            "quality_cost": summary["quality_cost"],
            "e2e_prefill_mean_ms": result["mean_ms"],
            "e2e_prefill_median_ms": result["median_ms"],
            "e2e_prefill_min_ms": result["min_ms"],
            "e2e_prefill_max_ms": result["max_ms"],
            "e2e_prefill_times_ms": result["times_ms"],
            "replaced_linear_count": report.replaced_linear_count,
            "skipped_linear_count": report.skipped_linear_count,
            "backend_counts": dict(report.backend_counts),
            "runtime_skipped": report.skipped,
            "requested_gpu": args.gpu,
            "local_gpu": local_gpu,
            "iters": args.iters,
            "warmup_iters": args.warmup_iters,
        }
        rows.append(row)
        write_csv(args.output_root / "validation" / "e2e_points" / f"point_{point_index:03d}.csv", [row])
        if not args.point_output_only:
            write_csv(args.output_root / "validation" / "pareto_e2e_validation.csv", rows)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if not args.point_output_only:
        write_json(
            args.output_root / "validation" / "pareto_e2e_validation_metadata.json",
            {
                "model_path": MODEL_PATH,
                "scenario": SCENARIO,
                "gpu": args.gpu,
                "dtype": args.dtype,
                "attn": args.attn,
                "points": point_indices,
                "iters": args.iters,
                "warmup_iters": args.warmup_iters,
            },
        )
    print(f"wrote {len(rows)} e2e validation rows")


def load_model(*, dtype: torch.dtype, attn: str, device: str):
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def benchmark_prefill(model, *, device: str, warmup_iters: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup_iters):
        ids = torch.randint(0, 1000, (1, 32), device=device)
        _ = model(ids, use_cache=False)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        input_ids = torch.randint(0, 1000, (SCENARIO["batch_size"], SCENARIO["input_tokens"]), device=device)
        start.record()
        _ = model(input_ids, use_cache=False)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {
        "times_ms": ";".join(f"{value:.6f}" for value in times),
        "mean_ms": mean(times),
        "median_ms": median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def select_point_indices(args: argparse.Namespace) -> list[int]:
    if args.points == "validation":
        selected = read_csv(args.output_root / "validation" / "selected_pareto_points.csv")
        return [int(f(row, "point_index")) for row in selected]
    return [int(item) for item in args.points.split(",") if item.strip()]


def find_pareto_policy(output_root: Path, point_index: int) -> Path:
    matches = sorted((output_root / "pareto" / "policies").glob(f"point_{point_index:03d}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no policy json for point {point_index}")
    return matches[0]


def local_cuda_index(requested_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA is required")
    if requested_gpu < count:
        return requested_gpu
    if visible:
        return 0
    raise RuntimeError(f"requested gpu {requested_gpu}, but torch sees {count} CUDA devices")


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
