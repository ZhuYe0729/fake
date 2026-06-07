#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_PATHS = {
    "llama2-7b": "/home/agent/wja/data/models/LLM-Research/llama-2-7b",
    "llama31-8b": "/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
}
MODEL_LABELS = {
    "llama2-7b": "Llama-2-7B",
    "llama31-8b": "Llama-3.1-8B",
}
POLICY_PATHS = {
    ("llama2-7b", "prefill_only"): "artifacts/results/benchmarks/hybrid/pred/llama_2_7b_prefill_only_policy.json",
    ("llama2-7b", "normal_01"): "artifacts/results/benchmarks/hybrid/pred/llama_2_7b_normal_01_policy.json",
    ("llama31-8b", "prefill_only"): "artifacts/results/benchmarks/hybrid/pred/llama_3_1_8b_prefill_only_policy.json",
    ("llama31-8b", "normal_01"): "artifacts/results/benchmarks/hybrid/pred/llama_3_1_8b_normal_01_policy.json",
}
SCENARIOS = {
    "prefill_only": {"batch_size": 16, "input_tokens": 1024, "output_tokens": 0},
    "normal_01": {"batch_size": 1, "input_tokens": 16384, "output_tokens": 32},
}
MANUAL_HYBRID = {
    ("llama2-7b", "prefill_only"): (413.9049, 2.1945),
    ("llama31-8b", "prefill_only"): (405.3724, 2.4285),
    ("llama2-7b", "normal_01"): (1930.0, 1.26),
    ("llama31-8b", "normal_01"): (2002.0, 1.13),
}
OUTPUT_CSV = REPO_ROOT / "artifacts/results/benchmarks/hybrid/pred/llama_predictor_hybrid_full_e2e.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full Llama predictor-hybrid E2E benchmark.")
    parser.add_argument("--model", choices=MODEL_PATHS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--policy-json", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.gpu)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    scenario = SCENARIOS[args.scenario]
    model_path = args.model_path or MODEL_PATHS[args.model]
    policy_path = args.policy_json or POLICY_PATHS[(args.model, args.scenario)]

    print(f"Loading {MODEL_LABELS[args.model]} from {model_path}")
    model = load_model(model_path, dtype=dtype, attn=args.attn, device=f"cuda:{args.gpu}")
    print(f"Applying predictor policy: {policy_path}")
    from fake.models.llama_kernels import replace_linear_with_llama_predictor_hybrid

    report = replace_linear_with_llama_predictor_hybrid(model, policy_path=policy_path, activation_dtype=dtype)
    print(f"replacement: replaced={report.replaced_linear_count} skipped={report.skipped_linear_count} counts={report.backend_counts}")
    if report.skipped:
        print(f"skipped[:5]={report.skipped[:5]}")

    result = benchmark(
        model,
        batch_size=scenario["batch_size"],
        input_tokens=scenario["input_tokens"],
        output_tokens=scenario["output_tokens"],
        warmup_iters=args.warmup_iters,
        device=f"cuda:{args.gpu}",
    )
    e2e = result["prefill_ms"] + scenario["output_tokens"] * result["decode_avg_ms"]
    manual_ms, manual_speedup = MANUAL_HYBRID[(args.model, args.scenario)]
    row = {
        "model": MODEL_LABELS[args.model],
        "scenario": args.scenario,
        "batch_size": scenario["batch_size"],
        "input_tokens": scenario["input_tokens"],
        "output_tokens": scenario["output_tokens"],
        "method": "predictor_hybrid",
        "prefill_ms": f"{result['prefill_ms']:.4f}",
        "decode_avg_ms": f"{result['decode_avg_ms']:.4f}",
        "decode_x_n_ms": f"{scenario['output_tokens'] * result['decode_avg_ms']:.4f}",
        "e2e_ms": f"{e2e:.4f}",
        "manual_hybrid_ms": f"{manual_ms:.4f}",
        "manual_hybrid_speedup_vs_dense_bf16": f"{manual_speedup:.4f}",
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "backend_counts": dict(report.backend_counts),
        "policy_json": str(policy_path),
    }
    write_result(row, append=args.append)
    print(f"result: prefill={row['prefill_ms']} decode_x_n={row['decode_x_n_ms']} e2e={row['e2e_ms']}")
    print(f"wrote {OUTPUT_CSV}")


def load_model(model_path: str, *, dtype: torch.dtype, attn: str, device: str):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def benchmark(
    model,
    *,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    warmup_iters: int,
    device: str,
) -> dict[str, float]:
    for _ in range(warmup_iters):
        ids = torch.randint(0, 1000, (1, 16), device=device)
        _ = model(ids, use_cache=False)
    torch.cuda.synchronize()

    input_ids = torch.randint(0, 1000, (batch_size, input_tokens), device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = model(input_ids, use_cache=output_tokens > 0)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = float(start.elapsed_time(end))

    if output_tokens == 0:
        return {
            "prefill_ms": prefill_ms,
            "decode_avg_ms": 0.0,
            "decode_first_ms": 0.0,
            "decode_steady_ms": 0.0,
        }

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

    return {
        "prefill_ms": prefill_ms,
        "decode_avg_ms": sum(decode_times) / len(decode_times),
        "decode_first_ms": decode_times[0],
        "decode_steady_ms": sum(decode_times[2:]) / max(len(decode_times[2:]), 1),
    }


def write_result(row: dict[str, object], *, append: bool) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = OUTPUT_CSV.exists()
    mode = "a" if append and exists else "w"
    with OUTPUT_CSV.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
