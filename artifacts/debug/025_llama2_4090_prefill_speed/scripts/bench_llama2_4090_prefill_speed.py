#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[4]
CUTLASS_WRAPPER_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
for path in (REPO_ROOT, CUTLASS_WRAPPER_ROOT, CUTLASS_WRAPPER_ROOT / "modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

from fake.kernels.offline_hybrid_policy import (  # noqa: E402
    POLICY_FORMAT,
    HybridPolicy,
    LayerPolicyDecision,
    ScenarioSpec,
    save_policy_json,
    write_policy_csv,
)
from fake.models.llama_kernels import replace_linear_with_llama_predictor_hybrid  # noqa: E402
from fake.models.qwen3_5_kernels import _build_manual_backend_module  # noqa: E402


MODEL_LABEL = "Llama-2-7B"
METHODS = ("dense_bf16", "sparse_bf16", "marlin_nvfp4")
SCENARIO_PRESETS = {
    "prefill_only": {"batch_size": 16, "input_tokens": 1024, "output_tokens": 0},
    "decode_heavy": {"batch_size": 1, "input_tokens": 1024, "output_tokens": 256},
    "prefill_decode": {"batch_size": 1, "input_tokens": 16384, "output_tokens": 32},
}
DEFAULT_MODEL_PATH = "/data/home/scxj523/run/wja/data/models/LLM-Research/llama-2-7b"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/debug/025_llama2_4090_prefill_speed"


@dataclass(frozen=True)
class LinearGroup:
    name: str
    n: int
    k: int
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Llama2-7B 4090 speed across prefill/decode scenarios.")
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_PRESETS, default=list(SCENARIO_PRESETS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=0)
    parser.add_argument("--custom-scenario-name", default="custom")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--linear-warmup-iters", type=int, default=3)
    parser.add_argument("--linear-measure-iters", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--custom-scenario", action="store_true")
    parser.add_argument("--skip-full-model", action="store_true")
    parser.add_argument("--skip-linear-aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    policy_dir = results_dir / "method_policies"
    results_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.cuda.set_device(args.gpu)

    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Model: {args.model_path}")

    groups = enumerate_linear_groups(args.model_path)
    scenarios = selected_scenarios(args)

    all_full_rows: list[dict[str, Any]] = []
    all_linear_rows: list[dict[str, Any]] = []
    for scenario_name, scenario in scenarios:
        scenario_dir = results_dir / scenario_name
        scenario_policy_dir = policy_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_policy_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\nScenario={scenario_name}: batch={scenario.batch_size}, "
            f"input_tokens={scenario.input_tokens}, output_tokens={scenario.output_tokens}, "
            f"m_prefill={scenario.m_prefill}, m_decode={scenario.m_decode}"
        )
        policies = write_method_policies(args.methods, groups, scenario, scenario_policy_dir)

        full_rows: list[dict[str, Any]] = []
        if not args.skip_full_model:
            for method in args.methods:
                try:
                    full_rows.extend(run_full_model_method(args, method, policies.get(method), scenario_name, scenario))
                except Exception as exc:
                    print(f"[full-model] scenario={scenario_name} method={method} failed: {type(exc).__name__}: {exc}")
                    full_rows.append(full_model_error_row(args, method, policies.get(method), scenario_name, scenario, exc))
                    gc.collect()
                    torch.cuda.empty_cache()
            write_csv(scenario_dir / "full_model_raw.csv", full_rows)
            write_csv(scenario_dir / "full_model_summary.csv", summarize_full_model(full_rows))
            if scenario_name == "prefill_only":
                write_csv(results_dir / "full_model_prefill_raw.csv", full_rows)
                write_csv(results_dir / "full_model_prefill_summary.csv", summarize_full_model(full_rows))
            all_full_rows.extend(full_rows)

        linear_rows: list[dict[str, Any]] = []
        if not args.skip_linear_aggregate:
            for method in args.methods:
                linear_rows.extend(run_linear_aggregate(args, method, groups, scenario_name, scenario))
            write_csv(scenario_dir / "linear_summary.csv", linear_rows)
            if scenario_name == "prefill_only":
                write_csv(results_dir / "linear_prefill_summary.csv", linear_rows)
            all_linear_rows.extend(linear_rows)

    if all_full_rows:
        write_csv(results_dir / "full_model_raw.csv", all_full_rows)
        write_csv(results_dir / "full_model_summary.csv", summarize_full_model(all_full_rows))
    if all_linear_rows:
        write_csv(results_dir / "linear_summary.csv", all_linear_rows)

    write_run_summary(args, scenarios, all_full_rows, all_linear_rows)


def selected_scenarios(args: argparse.Namespace) -> list[tuple[str, ScenarioSpec]]:
    if args.custom_scenario:
        return [
            (
                args.custom_scenario_name,
                ScenarioSpec(
                    batch_size=int(args.batch_size),
                    input_tokens=int(args.input_tokens),
                    output_tokens=int(args.output_tokens),
                ),
            )
        ]
    return [
        (
            name,
            ScenarioSpec(
                batch_size=int(spec["batch_size"]),
                input_tokens=int(spec["input_tokens"]),
                output_tokens=int(spec["output_tokens"]),
            ),
        )
        for name, spec in SCENARIO_PRESETS.items()
        if name in set(args.scenarios)
    ]


def enumerate_linear_groups(model_path: Path) -> list[LinearGroup]:
    from accelerate import init_empty_weights
    from fake.compression.modules import select_compressible_modules
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    selected = select_compressible_modules(model, "llama")
    grouped: dict[tuple[str, int, int], int] = {}
    for info in selected:
        if info.kind != "linear":
            continue
        name = normalize_group_name(info.name)
        n = int(info.module.out_features)
        k = int(info.module.in_features)
        grouped[(name, n, k)] = grouped.get((name, n, k), 0) + 1
    del model
    return [LinearGroup(name, n, k, count) for (name, n, k), count in sorted(grouped.items())]


def normalize_group_name(name: str) -> str:
    return re.sub(r"^(model\.)?layers\.\d+\.", "", name)


def write_method_policies(
    methods: list[str],
    groups: list[LinearGroup],
    scenario: ScenarioSpec,
    policy_dir: Path,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for method in methods:
        if method == "dense_bf16":
            continue
        policy = uniform_policy(groups, scenario, method)
        json_path = policy_dir / f"{method}_policy.json"
        csv_path = policy_dir / f"{method}_policy.csv"
        save_policy_json(policy, json_path)
        write_policy_csv(policy, csv_path)
        out[method] = json_path
    return out


def uniform_policy(groups: list[LinearGroup], scenario: ScenarioSpec, method: str) -> HybridPolicy:
    decisions = [
        LayerPolicyDecision(
            name=group.name,
            n=group.n,
            k=group.k,
            count=group.count,
            selected_prefill_backend=method,
            selected_decode_backend=method,
            selected_total_ms=None,
            selected_prefill_ms=None,
            selected_decode_ms=None,
            selected_conversion_ms=0.0,
            strategy_candidates=[],
            prefill_candidates=[],
            decode_candidates=[],
            conversion_candidates=[],
            reason="uniform_method_policy",
        )
        for group in groups
    ]
    return HybridPolicy(
        policy_format=POLICY_FORMAT,
        scenario={
            "batch_size": scenario.batch_size,
            "input_tokens": scenario.input_tokens,
            "output_tokens": scenario.output_tokens,
            "m_prefill": scenario.m_prefill,
            "m_decode": scenario.m_decode,
        },
        kernels=list(METHODS),
        include_conversion_cost=True,
        modules=decisions,
    )


def run_full_model_method(
    args: argparse.Namespace,
    method: str,
    policy_path: Path | None,
    scenario_name: str,
    scenario: ScenarioSpec,
) -> list[dict[str, Any]]:
    print(f"\n[full-model] scenario={scenario_name} method={method}")
    model = load_model(args.model_path, args.gpu)
    report = None
    if method != "dense_bf16":
        if policy_path is None:
            raise RuntimeError(f"missing policy for method={method}")
        report = replace_linear_with_llama_predictor_hybrid(
            model,
            policy_path=policy_path,
            activation_dtype=torch.bfloat16,
        )
        print(
            "replacement:",
            f"replaced={report.replaced_linear_count}",
            f"skipped={report.skipped_linear_count}",
            f"backends={dict(report.backend_counts)}",
        )

    rows = time_full_model(
        model,
        method=method,
        scenario_name=scenario_name,
        scenario=scenario,
        gpu=args.gpu,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        report=report,
        policy_path=policy_path,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def full_model_error_row(
    args: argparse.Namespace,
    method: str,
    policy_path: Path | None,
    scenario_name: str,
    scenario: ScenarioSpec,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "model": MODEL_LABEL,
        "method": method,
        "scenario": scenario_name,
        "batch_size": scenario.batch_size,
        "input_tokens": scenario.input_tokens,
        "output_tokens": scenario.output_tokens,
        "m_prefill": scenario.m_prefill,
        "m_decode": scenario.m_decode,
        "iteration": "",
        "prefill_ms": "",
        "decode_total_ms": "",
        "decode_avg_ms": "",
        "decode_first_ms": "",
        "decode_steady_ms": "",
        "e2e_ms": "",
        "status": "error",
        "error_msg": f"{type(exc).__name__}:{exc}",
        "gpu": args.gpu,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "",
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "replaced_linear_count": "",
        "skipped_linear_count": "",
        "backend_counts": "",
        "policy_json": "" if policy_path is None else str(policy_path),
    }


def load_model(model_path: Path, gpu: int) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = model.to(f"cuda:{gpu}")
    model.eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def time_full_model(
    model: nn.Module,
    *,
    method: str,
    scenario_name: str,
    scenario: ScenarioSpec,
    gpu: int,
    warmup_iters: int,
    measure_iters: int,
    report: Any,
    policy_path: Path | None,
) -> list[dict[str, Any]]:
    device = f"cuda:{gpu}"
    for _ in range(warmup_iters):
        ids = torch.randint(0, 1000, (scenario.batch_size, min(32, scenario.input_tokens)), device=device)
        out = model(ids, use_cache=False)
        del out
    torch.cuda.synchronize()

    rows = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for iteration in range(measure_iters):
        input_ids = torch.randint(0, 1000, (scenario.batch_size, scenario.input_tokens), device=device)
        start.record()
        out = model(input_ids, use_cache=scenario.output_tokens > 0)
        end.record()
        torch.cuda.synchronize()
        prefill_ms = float(start.elapsed_time(end))
        decode_times = []
        if scenario.output_tokens > 0:
            past_key_values = out.past_key_values
            next_token = torch.randint(0, 1000, (scenario.batch_size, 1), device=device)
            for _ in range(scenario.output_tokens):
                start.record()
                out = model(next_token, past_key_values=past_key_values, use_cache=True)
                end.record()
                torch.cuda.synchronize()
                decode_times.append(float(start.elapsed_time(end)))
                past_key_values = out.past_key_values
                next_token = torch.randint(0, 1000, (scenario.batch_size, 1), device=device)
            del past_key_values, next_token
        decode_total_ms = sum(decode_times)
        decode_avg_ms = mean(decode_times)
        decode_first_ms = decode_times[0] if decode_times else 0.0
        decode_steady_values = decode_times[2:] if len(decode_times) > 2 else decode_times
        decode_steady_ms = mean(decode_steady_values)
        e2e_ms = prefill_ms + decode_total_ms
        del out, input_ids
        rows.append(
            {
                "model": MODEL_LABEL,
                "method": method,
                "scenario": scenario_name,
                "batch_size": scenario.batch_size,
                "input_tokens": scenario.input_tokens,
                "output_tokens": scenario.output_tokens,
                "m_prefill": scenario.m_prefill,
                "m_decode": scenario.m_decode,
                "iteration": iteration,
                "prefill_ms": prefill_ms,
                "decode_total_ms": decode_total_ms,
                "decode_avg_ms": decode_avg_ms,
                "decode_first_ms": decode_first_ms,
                "decode_steady_ms": decode_steady_ms,
                "e2e_ms": e2e_ms,
                "gpu": gpu,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(gpu),
                "warmup_iters": warmup_iters,
                "measure_iters": measure_iters,
                "replaced_linear_count": "" if report is None else report.replaced_linear_count,
                "skipped_linear_count": "" if report is None else report.skipped_linear_count,
                "backend_counts": "" if report is None else json.dumps(dict(report.backend_counts), sort_keys=True),
                "policy_json": "" if policy_path is None else str(policy_path),
            }
        )
        print(
            f"  iter={iteration} prefill_ms={prefill_ms:.4f} "
            f"decode_total_ms={decode_total_ms:.4f} e2e_ms={e2e_ms:.4f}"
        )
    return rows


def summarize_full_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario_method.setdefault((str(row["scenario"]), str(row["method"])), []).append(row)
    out = []
    scenarios = sorted({scenario for scenario, _ in by_scenario_method})
    for scenario_name in scenarios:
        dense_rows = by_scenario_method.get((scenario_name, "dense_bf16"), [])
        dense_prefill = mean([float(row["prefill_ms"]) for row in dense_rows if row.get("prefill_ms") != ""])
        dense_decode = mean([float(row["decode_total_ms"]) for row in dense_rows if row.get("decode_total_ms") != ""])
        dense_e2e = mean([float(row["e2e_ms"]) for row in dense_rows if row.get("e2e_ms") != ""])
        for method in METHODS:
            method_rows = by_scenario_method.get((scenario_name, method), [])
            if not method_rows:
                continue
            out.append(summarize_full_model_method(method_rows, dense_prefill, dense_decode, dense_e2e))
    return out


def summarize_full_model_method(
    method_rows: list[dict[str, Any]],
    dense_prefill: float,
    dense_decode: float,
    dense_e2e: float,
) -> dict[str, Any]:
    first = method_rows[0]
    prefill_values = [float(row["prefill_ms"]) for row in method_rows if row.get("prefill_ms") != ""]
    decode_values = [float(row["decode_total_ms"]) for row in method_rows if row.get("decode_total_ms") != ""]
    decode_avg_values = [float(row["decode_avg_ms"]) for row in method_rows if row.get("decode_avg_ms") != ""]
    e2e_values = [float(row["e2e_ms"]) for row in method_rows if row.get("e2e_ms") != ""]
    if not e2e_values:
        return {
            "model": MODEL_LABEL,
            "method": first["method"],
            "scenario": first["scenario"],
            "batch_size": first["batch_size"],
            "input_tokens": first["input_tokens"],
            "output_tokens": first["output_tokens"],
            "m_prefill": first["m_prefill"],
            "m_decode": first.get("m_decode", ""),
            "status": "error",
            "error_msg": first.get("error_msg", "no successful timing rows"),
            "prefill_mean_ms": "",
            "decode_total_mean_ms": "",
            "decode_avg_mean_ms": "",
            "e2e_mean_ms": "",
            "measure_iters": 0,
            "speedup_vs_dense_prefill": "",
            "speedup_vs_dense_decode_total": "",
            "speedup_vs_dense_e2e": "",
            "replaced_linear_count": first.get("replaced_linear_count", ""),
            "skipped_linear_count": first.get("skipped_linear_count", ""),
            "backend_counts": first.get("backend_counts", ""),
            "policy_json": first.get("policy_json", ""),
        }
    prefill_mean = mean(prefill_values)
    decode_total_mean = mean(decode_values)
    decode_avg_mean = mean(decode_avg_values)
    e2e_mean = mean(e2e_values)
    return {
        "model": MODEL_LABEL,
        "method": first["method"],
        "scenario": first["scenario"],
        "status": "pass",
        "error_msg": "",
        "batch_size": first["batch_size"],
        "input_tokens": first["input_tokens"],
        "output_tokens": first["output_tokens"],
        "m_prefill": first["m_prefill"],
        "m_decode": first.get("m_decode", ""),
        "prefill_mean_ms": prefill_mean,
        "prefill_min_ms": min(prefill_values),
        "prefill_max_ms": max(prefill_values),
        "decode_total_mean_ms": decode_total_mean,
        "decode_avg_mean_ms": decode_avg_mean,
        "e2e_mean_ms": e2e_mean,
        "e2e_min_ms": min(e2e_values),
        "e2e_max_ms": max(e2e_values),
        "measure_iters": len(e2e_values),
        "speedup_vs_dense_prefill": "" if dense_prefill <= 0 else dense_prefill / prefill_mean,
        "speedup_vs_dense_decode_total": "" if dense_decode <= 0 or decode_total_mean <= 0 else dense_decode / decode_total_mean,
        "speedup_vs_dense_e2e": "" if dense_e2e <= 0 else dense_e2e / e2e_mean,
        "replaced_linear_count": first.get("replaced_linear_count", ""),
        "skipped_linear_count": first.get("skipped_linear_count", ""),
        "backend_counts": first.get("backend_counts", ""),
        "policy_json": first.get("policy_json", ""),
    }


def run_linear_aggregate(
    args: argparse.Namespace,
    method: str,
    groups: list[LinearGroup],
    scenario_name: str,
    scenario: ScenarioSpec,
) -> list[dict[str, Any]]:
    print(f"\n[linear-aggregate] scenario={scenario_name} method={method}")
    rows = []
    prefill_total = 0.0
    decode_total = 0.0
    fallback_count = 0
    for index, group in enumerate(groups):
        row = benchmark_linear_group(
            method,
            group,
            scenario,
            gpu=args.gpu,
            warmup_iters=args.linear_warmup_iters,
            measure_iters=args.linear_measure_iters,
            seed=3000 + index,
        )
        if not row["supported"] and method != "dense_bf16":
            failed_reason = row["reason"]
            row = benchmark_linear_group(
                "dense_bf16",
                group,
                scenario,
                gpu=args.gpu,
                warmup_iters=args.linear_warmup_iters,
                measure_iters=args.linear_measure_iters,
                seed=3000 + index,
            )
            row["method"] = method
            row["backend"] = "dense_bf16"
            row["used_fallback"] = True
            row["fallback_reason"] = failed_reason
            fallback_count += group.count
        weighted_prefill = float(row["weighted_prefill_ms"]) if row["supported"] else 0.0
        weighted_decode = float(row["weighted_decode_total_ms"]) if row["supported"] else 0.0
        prefill_total += weighted_prefill
        decode_total += weighted_decode
        row.update(
            {
                "model": MODEL_LABEL,
                "scenario": scenario_name,
                "batch_size": scenario.batch_size,
                "input_tokens": scenario.input_tokens,
                "output_tokens": scenario.output_tokens,
                "m_prefill": scenario.m_prefill,
                "m_decode": scenario.m_decode,
            }
        )
        rows.append(row)
        print(
            f"  {group.name:24s} n={group.n:5d} k={group.k:5d} x{group.count:2d} "
            f"-> prefill={weighted_prefill:.4f} decode={weighted_decode:.4f} ms"
        )

    rows.append(
        {
            "model": MODEL_LABEL,
            "method": method,
            "scenario": scenario_name,
            "linear_group": "__TOTAL__",
            "batch_size": scenario.batch_size,
            "input_tokens": scenario.input_tokens,
            "output_tokens": scenario.output_tokens,
            "m_prefill": scenario.m_prefill,
            "m_decode": scenario.m_decode,
            "weighted_prefill_ms": prefill_total,
            "weighted_decode_total_ms": decode_total,
            "weighted_e2e_ms": prefill_total + decode_total,
            "fallback_linear_count": fallback_count,
        }
    )
    return rows


def benchmark_linear_group(
    method: str,
    group: LinearGroup,
    scenario: ScenarioSpec,
    *,
    gpu: int,
    warmup_iters: int,
    measure_iters: int,
    seed: int,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{gpu}")
    base = make_base_linear(group.n, group.k, device, seed)
    try:
        module = make_backend_module(method, base).eval()
        x = torch.randn((1, scenario.m_prefill, group.k), device=device, dtype=torch.bfloat16)
        for _ in range(warmup_iters):
            y = module(x)
            assert_finite(y)
            del y
        torch.cuda.synchronize()
        prefill_ms = time_cuda(lambda: module(x), measure_iters)
        decode_ms = 0.0
        if scenario.output_tokens > 0:
            x_decode = torch.randn((1, scenario.m_decode, group.k), device=device, dtype=torch.bfloat16)
            for _ in range(warmup_iters):
                y = module(x_decode)
                assert_finite(y)
                del y
            torch.cuda.synchronize()
            decode_ms = time_cuda(lambda: module(x_decode), measure_iters)
            del x_decode
        weighted_prefill = group.count * prefill_ms
        weighted_decode = group.count * scenario.output_tokens * decode_ms
        return {
            "method": method,
            "backend": method,
            "linear_group": group.name,
            "n": group.n,
            "k": group.k,
            "count": group.count,
            "supported": True,
            "reason": "",
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "weighted_prefill_ms": weighted_prefill,
            "weighted_decode_total_ms": weighted_decode,
            "weighted_e2e_ms": weighted_prefill + weighted_decode,
            "used_fallback": False,
            "fallback_reason": "",
            "warmup_iters": warmup_iters,
            "measure_iters": measure_iters,
        }
    except Exception as exc:
        return {
            "method": method,
            "backend": method,
            "linear_group": group.name,
            "n": group.n,
            "k": group.k,
            "count": group.count,
            "supported": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "prefill_ms": "",
            "decode_ms": "",
            "weighted_prefill_ms": "",
            "weighted_decode_total_ms": "",
            "weighted_e2e_ms": "",
            "used_fallback": False,
            "fallback_reason": "",
            "warmup_iters": warmup_iters,
            "measure_iters": measure_iters,
        }
    finally:
        del base
        gc.collect()
        torch.cuda.empty_cache()


def make_backend_module(method: str, linear: nn.Linear) -> nn.Module:
    backend = "dense_bf16" if method == "dense_bf16" else method
    return _build_manual_backend_module(linear, backend, activation_dtype=torch.bfloat16)


def make_base_linear(n: int, k: int, device: torch.device, seed: int) -> nn.Linear:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    linear = nn.Linear(k, n, bias=False, device=device, dtype=torch.bfloat16)
    linear.weight.data.normal_(mean=0.0, std=0.02, generator=generator)
    linear.eval()
    linear.requires_grad_(False)
    return linear


def time_cuda(fn: Callable[[], torch.Tensor], iters: int) -> float:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        assert_finite(result)
        times.append(float(start.elapsed_time(end)))
        del result
    return sum(times) / len(times)


def assert_finite(tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all().item():
        raise RuntimeError("output contains NaN/Inf")


def write_run_summary(
    args: argparse.Namespace,
    scenarios: list[tuple[str, ScenarioSpec]],
    full_rows: list[dict[str, Any]],
    linear_rows: list[dict[str, Any]],
) -> None:
    summary_dir = args.output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Llama2-7B 4090 Prefill/Decode Speed Summary",
        "",
        f"- Model path: `{args.model_path}`",
        "- Scenarios:",
        *[
            f"  - `{name}`: `batch_size={scenario.batch_size}, input_tokens={scenario.input_tokens}, "
            f"output_tokens={scenario.output_tokens}, m_prefill={scenario.m_prefill}, m_decode={scenario.m_decode}`"
            for name, scenario in scenarios
        ],
        f"- Methods: `{', '.join(args.methods)}`",
        f"- Full-model rows: `{len(full_rows)}`",
        f"- Linear aggregate rows: `{len(linear_rows)}`",
        "",
        "Primary result: `results/full_model_summary.csv`.",
    ]
    (summary_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
