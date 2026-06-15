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

from common_pareto import (
    DEBUG_ROOT,
    METHODS,
    SCENARIO,
    DECODE_METHOD_MAP,
    f,
    is_legal_strategy,
    read_csv,
    read_json,
    write_csv,
    write_json,
)

FAKE_ROOT = Path(__file__).resolve().parents[4]
if str(FAKE_ROOT) not in sys.path:
    sys.path.insert(0, str(FAKE_ROOT))

from fake.kernels.offline_hybrid_policy import (  # noqa: E402
    HybridPolicy,
    LayerPolicyDecision,
    save_policy_json,
)

MODEL_PATH = "/home/agent/wja/data/models/LLM-Research/llama-2-7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate selected Pareto policies with real full-model prefill+decode E2E latency."
    )
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--points", default="0,3,7", help="Comma-separated point indices.")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--attn", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        model = load_model(dtype=dtype, attn=args.attn, device=device)
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
        e2e_total = result["prefill_ms"] + SCENARIO["output_tokens"] * result["decode_avg_ms"]
        row = {
            "point_index": point_index,
            "policy_json": str(policy_path),
            "converted_policy_json": str(converted_path),
            "predicted_total_latency_ms": summary["latency_ms"],
            "predicted_prefill_latency_ms": summary.get("total_prefill_ms", ""),
            "predicted_decode_latency_ms": summary.get("total_decode_ms", ""),
            "predicted_conversion_latency_ms": summary.get("total_conversion_ms", ""),
            "quality_cost": summary["quality_cost"],
            "e2e_prefill_ms": result["prefill_ms"],
            "e2e_decode_avg_ms": result["decode_avg_ms"],
            "e2e_decode_first_ms": result["decode_first_ms"],
            "e2e_decode_steady_ms": result["decode_steady_ms"],
            "e2e_total_ms": e2e_total,
            "e2e_times_ms": result.get("times_ms", ""),
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
        "e2e_prefill_ms": "",
        "e2e_decode_avg_ms": "",
        "e2e_decode_first_ms": "",
        "e2e_decode_steady_ms": "",
        "e2e_total_ms": "",
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
    for item in payload["modules"]:
        method = item["selected_prefill_backend"]
        prefill_backend = "dense_nvfp4" if method == "dense_nvfp4_prefill_marlin_decode" else method
        decode_backend = DECODE_METHOD_MAP.get(method, method)
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
                reason="quality_constrained_pareto_normal_01",
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
    return out


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

    times = []
    for _ in range(iters):
        input_ids = torch.randint(0, 1000, (batch_size, input_tokens), device=device)
        start.record()
        out = model(input_ids, use_cache=(output_tokens > 0))
        end.record()
        torch.cuda.synchronize()
        prefill_ms = float(start.elapsed_time(end))

        if output_tokens == 0:
            times.append(f"{prefill_ms:.6f}")
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
        times.append(f"{total_ms:.6f}")

    if output_tokens == 0:
        return {
            "times_ms": ";".join(times),
            "prefill_ms": prefill_ms,
            "decode_avg_ms": 0.0,
            "decode_first_ms": 0.0,
            "decode_steady_ms": 0.0,
        }

    return {
        "times_ms": ";".join(times),
        "prefill_ms": prefill_ms,
        "decode_avg_ms": sum(decode_times) / len(decode_times),
        "decode_first_ms": decode_times[0],
        "decode_steady_ms": sum(decode_times[2:]) / max(len(decode_times[2:]), 1),
    }


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
