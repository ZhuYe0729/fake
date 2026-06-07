#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.offline_hybrid_policy import ScenarioSpec
from scripts.run_main_hybrid_policy_retest import SCENARIOS, make_base_linear, make_candidate_module


CANDIDATES = (
    "dense_bf16",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)

GROUPS = {
    "mlp.down_proj": (4096, 11008, 32),
    "mlp.gate_proj": (11008, 4096, 32),
    "mlp.up_proj": (11008, 4096, 32),
    "self_attn.o_proj": (4096, 4096, 32),
    "self_attn.q_proj": (4096, 4096, 32),
}


@dataclass(frozen=True)
class GroupSpec:
    name: str
    n: int
    k: int
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/debug/005_llama2_warm_group_microbench/results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--prefill-iters", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    scenario = ScenarioSpec(**SCENARIOS["normal_02"])
    device = torch.device(f"cuda:{args.gpu}")

    rows = []
    for group_name in args.groups:
        n, k, count = GROUPS[group_name]
        group = GroupSpec(group_name, n, k, count)
        for candidate in CANDIDATES:
            try:
                row = bench_candidate_group(args, scenario, group, candidate, device)
            except Exception as exc:
                row = {
                    "group": group.name,
                    "candidate": candidate,
                    "supported": False,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            rows.append(row)
            gc.collect()
    write_csv(args.out_dir / "warm_group_microbench.csv", rows)
    write_rankings(args.out_dir / "warm_group_ranking.csv", rows)


def bench_candidate_group(args: argparse.Namespace, scenario: ScenarioSpec, group: GroupSpec, candidate: str, device: torch.device) -> dict[str, object]:
    modules: list[nn.Module] = []
    for index in range(group.count):
        base = make_base_linear(group.n, group.k, device, seed=5000 + index)
        module, prefill_backend, decode_backend = make_candidate_module(candidate, base, scenario)
        modules.append(module.eval())
        del base

    x_warm = torch.randn((1, min(32, int(scenario.input_tokens)), group.k), device=device, dtype=torch.bfloat16)
    x_prefill = torch.randn((1, int(scenario.input_tokens), group.k), device=device, dtype=torch.bfloat16)
    x_decode = torch.randn((1, 1, group.k), device=device, dtype=torch.bfloat16)

    for _ in range(args.warmup_iters):
        run_group(modules, x_warm)
    torch.cuda.synchronize()

    prefill_ms = time_cuda(lambda: run_group(modules, x_prefill), args.prefill_iters)
    decode_first_ms = time_cuda(lambda: run_group(modules, x_decode), 1)
    decode_steady_ms = time_decode_steps(modules, x_decode, args.decode_steps)
    projected_ms = prefill_ms + decode_first_ms + max(int(scenario.output_tokens) - 1, 0) * decode_steady_ms

    # One lightweight correctness check outside the timed loop.
    y = modules[0](x_decode)
    if not torch.isfinite(y.float()).all().item():
        raise RuntimeError("non-finite output")
    del y

    del modules, x_warm, x_prefill, x_decode
    return {
        "group": group.name,
        "candidate": candidate,
        "prefill_backend": prefill_backend,
        "decode_backend": decode_backend,
        "supported": True,
        "reason": "",
        "prefill_group_ms": prefill_ms,
        "decode_first_group_ms": decode_first_ms,
        "decode_steady_group_ms": decode_steady_ms,
        "projected_total_ms": projected_ms,
    }


def run_group(modules: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    out = None
    for module in modules:
        out = module(x)
    assert out is not None
    return out


def time_cuda(fn: Callable[[], torch.Tensor], iters: int) -> float:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
        del result
    return sum(times) / len(times)


def time_decode_steps(modules: list[nn.Module], x: torch.Tensor, steps: int) -> float:
    values = []
    for _ in range(steps):
        values.append(time_cuda(lambda: run_group(modules, x), 1))
    return sum(values) / len(values)


def write_rankings(path: Path, rows: list[dict[str, object]]) -> None:
    ranked = []
    by_group: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row.get("supported") is True:
            by_group.setdefault(str(row["group"]), []).append(row)
    for group, items in by_group.items():
        for rank, row in enumerate(sorted(items, key=lambda item: float(item["projected_total_ms"])), start=1):
            ranked.append(
                {
                    "group": group,
                    "rank": rank,
                    "candidate": row["candidate"],
                    "projected_total_ms": row["projected_total_ms"],
                    "prefill_group_ms": row["prefill_group_ms"],
                    "decode_first_group_ms": row["decode_first_group_ms"],
                    "decode_steady_group_ms": row["decode_steady_group_ms"],
                }
            )
    write_csv(path, ranked)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
