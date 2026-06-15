#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch

from common_pareto import DEBUG_ROOT, METHODS, SCENARIO, linear_groups_from_quality, load_module_quality_rows, write_csv

RETEST_SCRIPT = Path(__file__).resolve().parents[4] / "scripts/run_main_hybrid_policy_retest.py"
if str(RETEST_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(RETEST_SCRIPT.parent))

from run_main_hybrid_policy_retest import benchmark_manual_candidate  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real-kernel Llama2 prefill-only linear latency.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--manual-warmup", type=int, default=3)
    parser.add_argument("--manual-iters", type=int, default=10)
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--max-groups", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_gpu = local_cuda_index(args.gpu)
    torch.cuda.set_device(local_gpu)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []
    quality_rows = load_module_quality_rows()
    groups = linear_groups_from_quality(quality_rows)
    if args.max_groups:
        groups = groups[: args.max_groups]

    scenario = SimpleScenario(
        batch_size=SCENARIO["batch_size"],
        input_tokens=SCENARIO["input_tokens"],
        output_tokens=SCENARIO["output_tokens"],
    )
    bench_args = argparse.Namespace(
        gpu=local_gpu,
        manual_warmup=args.manual_warmup,
        manual_iters=args.manual_iters,
    )
    for group_index, group in enumerate(groups):
        for method in methods:
            print(f"benchmark group={group.name} n={group.n} k={group.k} count={group.count} method={method}")
            row = benchmark_manual_candidate(bench_args, group, scenario, method, seed=8000 + group_index)
            row.update(
                {
                    "model_key": "llama2-7b",
                    "scenario": "prefill_only",
                    "linear_group": group.name,
                    "n": group.n,
                    "k": group.k,
                    "count": group.count,
                    "method": method,
                    "source": "fresh_microbench",
                    "requested_gpu": args.gpu,
                    "local_gpu": local_gpu,
                    "m_prefill": scenario.m_prefill,
                }
            )
            rows.append(row)
            gc.collect()
            torch.cuda.empty_cache()
            write_csv(args.output_root / "latency" / "prefill_latency.csv", rows)
    print(f"wrote {len(rows)} latency rows")


class SimpleScenario:
    def __init__(self, *, batch_size: int, input_tokens: int, output_tokens: int):
        self.batch_size = batch_size
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    @property
    def m_prefill(self) -> int:
        return int(self.batch_size) * int(self.input_tokens)

    @property
    def m_decode(self) -> int:
        return int(self.batch_size)


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
    main()
