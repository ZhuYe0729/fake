#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_main_hybrid_policy_retest import MODELS, SCENARIOS, apply_policy, benchmark_model, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["prefill_only", "normal_01", "normal_02"], required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method-family", default="single")
    parser.add_argument("--policy-or-method", default="dense_nvfp4_prefill_marlin_decode")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    dtype = torch.bfloat16
    model = load_model("llama2-7b", dtype=dtype, gpu=args.gpu)
    report = apply_policy("llama2-7b", model, args.policy, dtype)
    scenario = SCENARIOS[args.scenario]
    result = benchmark_model(model, scenario, args.gpu, warmup_iters=1)
    row = {
        "model": MODELS["llama2-7b"]["label"],
        "scenario": args.scenario,
        "method_family": args.method_family,
        "policy_or_method": args.policy_or_method,
        "batch_size": scenario["batch_size"],
        "input_tokens": scenario["input_tokens"],
        "output_tokens": scenario["output_tokens"],
        "timing_mode": "warm_e2e_aligned",
        "prefill_ms": result["prefill_ms"],
        "decode_avg_ms": result["decode_avg_ms"],
        "decode_first_ms": result["decode_first_ms"],
        "decode_steady_ms": result["decode_steady_ms"],
        "decode_x_n_ms": scenario["output_tokens"] * result["decode_avg_ms"],
        "e2e_ms": result["prefill_ms"] + scenario["output_tokens"] * result["decode_avg_ms"],
        "replaced_linear_count": report.replaced_linear_count,
        "skipped_linear_count": report.skipped_linear_count,
        "backend_counts": dict(report.backend_counts),
        "policy_json": str(args.policy),
    }
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
