#!/usr/bin/env python
from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear, SPARSE_BF16_BLOCKED_SHAPES
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear
from fake.kernels.offline_hybrid_policy import load_policy_json


OUT_DIR = REPO_ROOT / "artifacts/results/benchmarks/hybrid/pred"
POLICY_FILES = {
    ("Llama-2-7B", "prefill_only"): OUT_DIR / "llama_2_7b_prefill_only_policy.json",
    ("Llama-2-7B", "normal_01"): OUT_DIR / "llama_2_7b_normal_01_policy.json",
    ("Llama-3.1-8B", "prefill_only"): OUT_DIR / "llama_3_1_8b_prefill_only_policy.json",
    ("Llama-3.1-8B", "normal_01"): OUT_DIR / "llama_3_1_8b_normal_01_policy.json",
    ("Qwen3.5-9B", "prefill_only"): OUT_DIR / "qwen3_5_9b_prefill_only_policy.json",
    ("Qwen3.5-9B", "normal_01"): OUT_DIR / "qwen3_5_9b_normal_01_policy.json",
}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows = []
    summary_rows = []
    for (model, scenario), policy_path in POLICY_FILES.items():
        policy = load_policy_json(policy_path)
        scenario_spec = policy.scenario
        total_prefill = 0.0
        total_decode = 0.0
        print(f"\n== {model} {scenario} ==")
        for index, module in enumerate(policy.modules):
            prefill_backend = module.selected_prefill_backend
            decode_backend = module.selected_decode_backend
            if prefill_backend is None or decode_backend is None:
                continue
            prefill_ms = benchmark_one(
                backend=prefill_backend,
                m=int(scenario_spec["m_prefill"]),
                n=module.n,
                k=module.k,
                device=device,
                seed=1000 + index,
            )
            decode_ms = 0.0
            if int(scenario_spec["output_tokens"]) > 0:
                decode_ms = benchmark_one(
                    backend=decode_backend,
                    m=int(scenario_spec["m_decode"]),
                    n=module.n,
                    k=module.k,
                    device=device,
                    seed=2000 + index,
                )
            weighted_prefill = module.count * prefill_ms
            weighted_decode = module.count * int(scenario_spec["output_tokens"]) * decode_ms
            total_prefill += weighted_prefill
            total_decode += weighted_decode
            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "linear_group": module.name,
                    "count": module.count,
                    "n": module.n,
                    "k": module.k,
                    "m_prefill": scenario_spec["m_prefill"],
                    "m_decode": scenario_spec["m_decode"],
                    "output_tokens": scenario_spec["output_tokens"],
                    "prefill_backend": prefill_backend,
                    "decode_backend": decode_backend,
                    "prefill_ms": f"{prefill_ms:.6f}",
                    "decode_ms": f"{decode_ms:.6f}",
                    "weighted_prefill_ms": f"{weighted_prefill:.6f}",
                    "weighted_decode_ms": f"{weighted_decode:.6f}",
                    "weighted_e2e_ms": f"{weighted_prefill + weighted_decode:.6f}",
                }
            )
            print(
                f"  {module.name:28s} x{module.count:<3d} "
                f"pf={prefill_backend}:{prefill_ms:.4f}ms "
                f"dec={decode_backend}:{decode_ms:.4f}ms"
            )
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(
            {
                "model": model,
                "scenario": scenario,
                "linear_group": "__TOTAL__",
                "count": "",
                "n": "",
                "k": "",
                "m_prefill": scenario_spec["m_prefill"],
                "m_decode": scenario_spec["m_decode"],
                "output_tokens": scenario_spec["output_tokens"],
                "prefill_backend": "",
                "decode_backend": "",
                "prefill_ms": "",
                "decode_ms": "",
                "weighted_prefill_ms": f"{total_prefill:.6f}",
                "weighted_decode_ms": f"{total_decode:.6f}",
                "weighted_e2e_ms": f"{total_prefill + total_decode:.6f}",
            }
        )
        manual_ms, manual_speedup = manual_hybrid_result(model, scenario)
        summary_rows.append(
            {
                "model": model,
                "scenario": scenario,
                "gpu_policy_linear_prefill_ms": f"{total_prefill:.6f}",
                "gpu_policy_linear_decode_x_n_ms": f"{total_decode:.6f}",
                "gpu_policy_linear_e2e_ms": f"{total_prefill + total_decode:.6f}",
                "manual_hybrid_ms": "" if manual_ms is None else f"{manual_ms:.6f}",
                "manual_hybrid_speedup_vs_dense_bf16": "" if manual_speedup is None else f"{manual_speedup:.4f}",
            }
        )
        print(f"  TOTAL prefill={total_prefill:.3f}ms decode_x_n={total_decode:.3f}ms e2e={total_prefill + total_decode:.3f}ms")
    write_csv(OUT_DIR / "gpu_policy_module_e2e.csv", rows)
    write_csv(OUT_DIR / "gpu_policy_module_summary.csv", summary_rows)
    print(f"\nwrote {OUT_DIR / 'gpu_policy_module_e2e.csv'}")
    print(f"wrote {OUT_DIR / 'gpu_policy_module_summary.csv'}")


def benchmark_one(*, backend: str, m: int, n: int, k: int, device: torch.device, seed: int) -> float:
    base = make_base_linear(n, k, device, seed)
    module = make_module(backend, base, device)
    x = torch.randn((1, m, k), device=device, dtype=torch.bfloat16)
    latency = time_cuda(lambda: module(x), warmup=5, iters=20)
    del x, module, base
    gc.collect()
    torch.cuda.empty_cache()
    return latency


def make_base_linear(n: int, k: int, device: torch.device, seed: int) -> nn.Linear:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    linear = nn.Linear(k, n, bias=False, device=device, dtype=torch.bfloat16)
    linear.weight.data.normal_(mean=0.0, std=0.02, generator=generator)
    linear.eval()
    linear.requires_grad_(False)
    return linear


def make_module(backend: str, base_linear: nn.Linear, device: torch.device) -> nn.Module:
    wrapper = load_wrapper()
    if backend == "dense_bf16":
        return base_linear
    if backend == "dense_nvfp4":
        return wrapper.NVFP4Linear.from_linear(base_linear, device=device).eval()
    if backend == "marlin_nvfp4":
        return wrapper.MarlinNVFP4Linear.from_linear(base_linear, device=device, activation_dtype=torch.bfloat16).eval()
    if backend == "sparse_bf16":
        if (base_linear.out_features, base_linear.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            raise ValueError("sparse_bf16 blocked shape")
        sparse = wrapper.SparseBF16Linear.from_linear(base_linear, device=device, prune=True).eval()
        return PaddedSparseBF16Linear(sparse, 8).eval()
    if backend == "sparse_nvfp4":
        sparse = wrapper.SparseNVFP4Linear.from_linear(base_linear, device=device, prune=True).eval()
        return PaddedSparseNVFP4Linear(sparse, 32).eval()
    raise ValueError(f"unknown backend: {backend}")


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    del result
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        if not torch.isfinite(result.float()).all().item():
            raise RuntimeError("module output contains NaN/Inf")
        times.append(float(start.elapsed_time(end)))
        del result
    return sum(times) / len(times)


def load_wrapper():
    import importlib

    for module_name in ("fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper", "cutlass_wrapper"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass
    raise RuntimeError("CUTLASS wrapper package is not importable")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manual_hybrid_result(model: str, scenario: str) -> tuple[float | None, float | None]:
    if scenario == "normal_01":
        values = {
            "Llama-2-7B": (1930.0, 1.26),
            "Llama-3.1-8B": (2002.0, 1.13),
            "Qwen3.5-9B": (3308.0, 1.27),
        }
        return values[model]
    values = {
        "Llama-2-7B": (413.9049, 2.1945),
        "Llama-3.1-8B": (405.3724, 2.4285),
        "Qwen3.5-9B": (427.2405, 2.2766),
    }
    return values[model]


if __name__ == "__main__":
    main()
