#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any
from collections import Counter

import torch

from common_pareto import (
    DEBUG_ROOT,
    METHODS,
    SCENARIO,
    f,
    read_csv,
    read_json,
    write_csv,
    write_json,
)

FAKE_ROOT = Path(__file__).resolve().parents[4]
if str(FAKE_ROOT) not in sys.path:
    sys.path.insert(0, str(FAKE_ROOT))

from fake.kernels.offline_hybrid_policy import (
    HybridPolicy,
    LayerPolicyDecision,
    save_policy_json,
)

MODEL_PATHS = {
    "llama2-7b": "/home/agent/wja/data/models/LLM-Research/llama-2-7b",
    "llama31-8b": "/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate selected Pareto policies with real full-model prefill+decode E2E latency for normal_02."
    )
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--model", choices=MODEL_PATHS, default="llama31-8b")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--points", default="0,3,7", help="Comma-separated point indices.")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--attn", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root == DEBUG_ROOT and args.model != "llama2-7b":
        args.output_root = DEBUG_ROOT / "models" / args.model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    device = f"cuda:{local_gpu}"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    point_indices = [int(item) for item in args.points.split(",") if item.strip()]
    rows = []
    for point_index in point_indices:
        policy_path = find_pareto_policy(args.output_root, point_index)
        converted_path = convert_policy(args.output_root, policy_path, point_index)
        print(f"e2e validating point={point_index} policy={converted_path}")
        model = load_model(MODEL_PATHS[args.model], dtype=dtype, attn=args.attn, device=device)
        from fake.models.llama_kernels import replace_linear_with_llama_predictor_hybrid

        try:
            report = replace_linear_with_llama_predictor_hybrid(
                model, policy_path=converted_path, activation_dtype=dtype
            )
        except Exception as exc:
            rows.append(make_error_row(point_index, policy_path, converted_path, local_gpu, args, f"replace_failed: {exc}"))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        print(f"  replaced={report.replaced_linear_count} skipped={report.skipped_linear_count} backends={dict(report.backend_counts)}")
        if report.skipped_linear_count > 0:
            print(f"  skipped[:5]={report.skipped[:5]}")

        try:
            result = benchmark(model, device=device, warmup_iters=args.warmup_iters, iters=args.iters)
        except Exception as exc:
            rows.append(make_error_row(point_index, policy_path, converted_path, local_gpu, args, f"benchmark_failed: {exc}"))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        summary = read_json(policy_path)["summary"]
        e2e_times_str = ";".join(f"{t:.6f}" for t in result["total_times"])
        row = {
            "point_index": point_index,
            "policy_json": str(policy_path),
            "converted_policy_json": str(converted_path),
            "predicted_total_latency_ms": summary["latency_ms"],
            "predicted_prefill_latency_ms": summary.get("total_prefill_ms", ""),
            "predicted_decode_latency_ms": summary.get("total_decode_ms", ""),
            "predicted_conversion_latency_ms": summary.get("total_conversion_ms", ""),
            "quality_cost": summary["quality_cost"],
            "e2e_total_mean_ms": result["total_mean"],
            "e2e_total_median_ms": result["total_median"],
            "e2e_total_min_ms": result["total_min"],
            "e2e_total_max_ms": result["total_max"],
            "e2e_prefill_mean_ms": result["prefill_mean"],
            "e2e_decode_avg_mean_ms": result["decode_avg_mean"],
            "e2e_decode_first_mean_ms": result["decode_first_mean"],
            "e2e_decode_steady_mean_ms": result["decode_steady_mean"],
            "e2e_times_ms": e2e_times_str,
            "replaced_linear_count": report.replaced_linear_count,
            "skipped_linear_count": report.skipped_linear_count,
            "backend_counts": dict(report.backend_counts),
            "e2e_status": "ok",
            "unsupported_reason": "",
            "requested_gpu": args.gpu,
            "local_gpu": local_gpu,
            "iters": args.iters,
            "warmup_iters": args.warmup_iters,
        }
        rows.append(row)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(args.output_root / "validation" / "pareto_e2e_validation.csv", rows)

    write_json(
        args.output_root / "validation" / "pareto_e2e_validation_metadata.json",
        {
            "model": args.model,
            "model_path": MODEL_PATHS[args.model],
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


def make_error_row(
    point_index: int,
    policy_path: Path,
    converted_path: Path,
    local_gpu: int,
    args: argparse.Namespace,
    reason: str,
) -> dict[str, Any]:
    return {
        "point_index": point_index,
        "policy_json": str(policy_path),
        "converted_policy_json": str(converted_path),
        "predicted_total_latency_ms": "",
        "predicted_prefill_latency_ms": "",
        "predicted_decode_latency_ms": "",
        "predicted_conversion_latency_ms": "",
        "quality_cost": "",
        "e2e_total_mean_ms": "",
        "e2e_total_median_ms": "",
        "e2e_total_min_ms": "",
        "e2e_total_max_ms": "",
        "e2e_prefill_mean_ms": "",
        "e2e_decode_avg_mean_ms": "",
        "e2e_decode_first_mean_ms": "",
        "e2e_decode_steady_mean_ms": "",
        "e2e_times_ms": "",
        "replaced_linear_count": "",
        "skipped_linear_count": "",
        "backend_counts": {},
        "e2e_status": "error",
        "unsupported_reason": reason,
        "requested_gpu": args.gpu,
        "local_gpu": local_gpu,
        "iters": args.iters,
        "warmup_iters": args.warmup_iters,
    }


def convert_policy(output_root: Path, policy_path: Path, point_index: int) -> Path:
    payload = read_json(policy_path)
    decisions = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    for item in payload["modules"]:
        prefill_backend = item["selected_prefill_backend"]
        decode_backend = item["selected_decode_backend"]
        pair_counts[(prefill_backend, decode_backend)] += 1
        decisions.append(
            LayerPolicyDecision(
                name=item["module_name"],
                n=int(item["n"]),
                k=int(item["k"]),
                count=1,
                selected_prefill_backend=prefill_backend,
                selected_decode_backend=decode_backend,
                selected_total_ms=float(item["selected_total_ms"]),
                selected_prefill_ms=float(item.get("selected_prefill_ms", 0)),
                selected_decode_ms=float(item.get("selected_decode_ms", 0)),
                selected_conversion_ms=float(item.get("selected_conversion_ms", 0)),
                strategy_candidates=[],
                prefill_candidates=[],
                decode_candidates=[],
                conversion_candidates=[],
                reason="quality_constrained_pareto_normal_02",
            )
        )
    policy = HybridPolicy(
        policy_format="offline_hybrid_policy_v1",
        scenario={
            "batch_size": int(SCENARIO["batch_size"]),
            "input_tokens": int(SCENARIO["input_tokens"]),
            "output_tokens": int(SCENARIO["output_tokens"]),
            "m_prefill": int(SCENARIO["m_prefill"]),
            "m_decode": int(SCENARIO["m_decode"]),
        },
        kernels=["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"],
        include_conversion_cost=True,
        modules=decisions,
    )
    out = output_root / "validation" / "converted_policies" / f"point_{point_index:03d}_offline_hybrid_policy.json"
    save_policy_json(policy, out)
    validate_backend_pairs(point_index, pair_counts)
    return out


def validate_backend_pairs(point_index: int, pair_counts: Counter[tuple[str, str]]) -> None:
    print(f"  converted backend pairs={dict(pair_counts)}")


def load_model(model_path: str, *, dtype: torch.dtype, attn: str, device: str):
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def benchmark(model, *, device: str, warmup_iters: int, iters: int) -> dict[str, Any]:
    batch_size = SCENARIO["batch_size"]
    input_tokens = SCENARIO["input_tokens"]
    output_tokens = SCENARIO["output_tokens"]

    for _ in range(warmup_iters):
        ids = torch.randint(0, 1000, (1, 16), device=device)
        _ = model(ids, use_cache=False)
    torch.cuda.synchronize()

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

        if output_tokens == 0:
            total_times.append(prefill_ms)
            decode_avg_times.append(0.0)
            decode_first_times.append(0.0)
            decode_steady_times.append(0.0)
            continue

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

        total_ms = prefill_ms + sum(decode_times)
        total_times.append(total_ms)
        decode_avg_times.append(sum(decode_times) / len(decode_times))
        decode_first_times.append(decode_times[0])
        decode_steady_times.append(sum(decode_times[2:]) / max(len(decode_times[2:]), 1))

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


def find_pareto_policy(output_root: Path, point_index: int) -> Path:
    matches = sorted((output_root / "pareto" / "policies").glob(f"point_{point_index:03d}_*.json"))
    if not matches:
        raise FileNotFoundError(f"no policy json for point {point_index}")
    points = read_csv(output_root / "pareto" / "pareto_points.csv")
    expected = next((row for row in points if int(f(row, "point_index")) == point_index), None)
    if expected is None:
        raise FileNotFoundError(f"point {point_index} is not present in pareto_points.csv")
    expected_budget = f(expected, "quality_budget")
    exact = []
    for path in matches:
        payload = read_json(path)
        summary = payload.get("summary", {})
        if abs(float(summary.get("quality_budget", -1.0)) - expected_budget) < 1e-6:
            exact.append(path)
    if len(exact) != 1:
        raise RuntimeError(
            f"ambiguous policy json for point {point_index}: expected_budget={expected_budget}, "
            f"matches={[str(p) for p in matches]}, exact={[str(p) for p in exact]}"
        )
    return exact[0]


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
