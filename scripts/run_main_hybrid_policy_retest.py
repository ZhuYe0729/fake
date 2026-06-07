#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
CUTLASS_WRAPPER_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
for path in (REPO_ROOT, CUTLASS_WRAPPER_ROOT, CUTLASS_WRAPPER_ROOT / "modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from fake.kernels.cutlass_sparse_bf16 import PaddedSparseBF16Linear, SPARSE_BF16_BLOCKED_SHAPES
from fake.kernels.cutlass_sparse_nvfp4 import PaddedSparseNVFP4Linear
from fake.kernels.offline_hybrid_policy import (
    POLICY_FORMAT,
    HybridPolicy,
    KernelPredictionRecord,
    LayerPolicyDecision,
    ScenarioSpec,
    StrategyCandidate,
    save_policy_json,
    write_policy_csv,
)
from fake.models.qwen3_5 import qwen3_5_model_path
from fake.models.qwen3_5_kernels import QwenHybridDenseNVFP4Linear
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor


OUTPUT_ROOT = REPO_ROOT / "artifacts/results/main/002_warm_e2e_aligned_policy_retest"
TIMING_MODE = "warm_e2e_aligned"
KERNELS = ["dense_bf16", "sparse_bf16", "dense_nvfp4", "sparse_nvfp4", "marlin_nvfp4"]
MANUAL_CANDIDATES = KERNELS + ["dense_nvfp4_prefill_marlin_decode"]
SCENARIOS = {
    "prefill_only": {"batch_size": 16, "input_tokens": 1024, "output_tokens": 0},
    "normal_01": {"batch_size": 1, "input_tokens": 16384, "output_tokens": 32},
    "normal_02": {"batch_size": 1, "input_tokens": 16384, "output_tokens": 256},
}
MODELS = {
    "llama2-7b": {
        "label": "Llama-2-7B",
        "family": "llama",
        "path": "/home/agent/wja/data/models/LLM-Research/llama-2-7b",
    },
    "llama31-8b": {
        "label": "Llama-3.1-8B",
        "family": "llama",
        "path": "/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    },
    "qwen35-9b": {
        "label": "Qwen3.5-9B",
        "family": "qwen3_5",
        "path": str(qwen3_5_model_path("9B")),
    },
}


@dataclass(frozen=True)
class LinearGroup:
    name: str
    n: int
    k: int
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main hybrid policy retest.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--families", nargs="+", choices=["single", "manual", "pred"], default=["single", "manual", "pred"])
    parser.add_argument("--run-e2e", action="store_true", help="Run full-model E2E after generating policies.")
    parser.add_argument("--skip-existing-e2e", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--manual-warmup", type=int, default=3)
    parser.add_argument("--manual-iters", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_root_readme(args.output_root)
    torch.cuda.set_device(args.gpu)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    shape_cache: dict[str, list[LinearGroup]] = {}
    for model_key in args.models:
        shape_cache[model_key] = enumerate_linear_groups(model_key)

    for scenario_name in args.scenarios:
        scenario = ScenarioSpec(**SCENARIOS[scenario_name])
        if "single" in args.families:
            for method in KERNELS:
                for model_key in args.models:
                    groups = shape_cache[model_key]
                    out_dir = args.output_root / "single" / method / scenario_name
                    policy = single_policy(groups, scenario, method)
                    write_policy_outputs(out_dir, model_key, scenario_name, policy)
                    single_linear_summary(args, out_dir, model_key, scenario_name, groups, scenario, method)
                    if args.run_e2e:
                        run_full_e2e(args, model_key, scenario_name, out_dir, method="single", policy_path=out_dir / f"{model_key}_policy.json", dtype=dtype)
        if "manual" in args.families:
            for model_key in args.models:
                groups = shape_cache[model_key]
                out_dir = args.output_root / "manual" / scenario_name
                policy = manual_policy(args, model_key, scenario_name, groups, scenario, out_dir)
                write_policy_outputs(out_dir, model_key, scenario_name, policy)
                if args.run_e2e:
                    run_full_e2e(args, model_key, scenario_name, out_dir, method="manual", policy_path=out_dir / f"{model_key}_policy.json", dtype=dtype)
        if "pred" in args.families:
            predictor = KernelLatencyPredictor(model_root=DEFAULT_MODEL_ROOT, kernels=KERNELS)
            for model_key in args.models:
                groups = shape_cache[model_key]
                out_dir = args.output_root / "pred" / scenario_name
                policy = pred_policy(model_key, scenario_name, groups, scenario, predictor, out_dir)
                write_policy_outputs(out_dir, model_key, scenario_name, policy)
                if args.run_e2e:
                    run_full_e2e(args, model_key, scenario_name, out_dir, method="pred", policy_path=out_dir / f"{model_key}_policy.json", dtype=dtype)

    write_comparison(args.output_root)


def enumerate_linear_groups(model_key: str) -> list[LinearGroup]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from fake.compression.modules import select_compressible_modules

    config = AutoConfig.from_pretrained(MODELS[model_key]["path"], trust_remote_code=True, local_files_only=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    selected = select_compressible_modules(model, MODELS[model_key]["family"])
    grouped: dict[tuple[str, int, int], int] = {}
    for info in selected:
        if info.kind != "linear":
            continue
        name = normalize_group_name(info.name)
        n = int(info.module.out_features)
        k = int(info.module.in_features)
        grouped[(name, n, k)] = grouped.get((name, n, k), 0) + 1
    return [LinearGroup(name, n, k, count) for (name, n, k), count in sorted(grouped.items())]


def normalize_group_name(name: str) -> str:
    name = re.sub(r"^(model\.)?layers\.\d+\.", "", name)
    name = re.sub(r"^(model\.)?language_model\.layers\.\d+\.", "", name)
    return name


def single_policy(groups: list[LinearGroup], scenario: ScenarioSpec, method: str) -> HybridPolicy:
    decisions = [
        make_decision(
            group,
            selected_prefill=method,
            selected_decode=method,
            total_ms=None,
            prefill_ms=None,
            decode_ms=None,
            conversion_ms=0.0,
            candidates=[],
        )
        for group in groups
    ]
    return make_policy(scenario, decisions)


def manual_policy(
    args: argparse.Namespace,
    model_key: str,
    scenario_name: str,
    groups: list[LinearGroup],
    scenario: ScenarioSpec,
    out_dir: Path,
) -> HybridPolicy:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    decisions = []
    for index, group in enumerate(groups):
        candidates = []
        for candidate in MANUAL_CANDIDATES:
            row = benchmark_manual_candidate(args, group, scenario, candidate, seed=1000 + index)
            row.update({"model": MODELS[model_key]["label"], "scenario": scenario_name, "linear_group": group.name, "n": group.n, "k": group.k, "count": group.count})
            rows.append(row)
            if row["supported"]:
                candidates.append(row)
        best = min(candidates, key=lambda row: float(row["weighted_total_ms"]))
        decisions.append(
            make_decision(
                group,
                selected_prefill=best["prefill_backend"],
                selected_decode=best["decode_backend"],
                total_ms=float(best["weighted_total_ms"]),
                prefill_ms=float(best["prefill_ms"]),
                decode_ms=float(best["decode_ms"]),
                conversion_ms=float(best.get("online_conversion_ms", 0.0)),
                candidates=[],
            )
        )
    write_csv(out_dir / f"{model_key}_manual_candidates.csv", rows)
    write_linear_summary(out_dir / f"{model_key}_linear_summary.csv", MODELS[model_key]["label"], scenario_name, decisions)
    return make_policy(scenario, decisions)


def benchmark_manual_candidate(args: argparse.Namespace, group: LinearGroup, scenario: ScenarioSpec, candidate: str, *, seed: int) -> dict[str, Any]:
    device = torch.device(f"cuda:{args.gpu}")
    base = make_base_linear(group.n, group.k, device, seed)
    try:
        module, prefill_backend, decode_backend = make_candidate_module(candidate, base, scenario)
        warmup_m = max(int(scenario.batch_size) * min(int(scenario.input_tokens), 32), int(scenario.batch_size))
        x_warm = torch.randn((1, warmup_m, group.k), device=device, dtype=torch.bfloat16)
        x_prefill = torch.randn((1, scenario.m_prefill, group.k), device=device, dtype=torch.bfloat16)
        x_decode = torch.randn((1, scenario.m_decode, group.k), device=device, dtype=torch.bfloat16)
        for _ in range(args.manual_warmup):
            y = module(x_warm)
            assert_finite(y)
            del y
        torch.cuda.synchronize()
        prefill_first_ms = time_cuda(lambda: module(x_prefill), 1)
        prefill_steady_ms = time_cuda(lambda: module(x_prefill), args.manual_iters)
        prefill_ms = prefill_steady_ms
        decode_ms = 0.0
        decode_first_ms = 0.0
        decode_steady_ms = 0.0
        if scenario.output_tokens > 0:
            decode_first_ms = time_cuda(lambda: module(x_decode), 1)
            decode_steady_ms = time_cuda(lambda: module(x_decode), args.manual_iters)
            decode_ms = (decode_first_ms + (int(scenario.output_tokens) - 1) * decode_steady_ms) / int(scenario.output_tokens)
        total_ms = prefill_ms + decode_first_ms + max(int(scenario.output_tokens) - 1, 0) * decode_steady_ms
        weighted = manual_group_total_warm_e2e_ms(
            group,
            scenario,
            candidate,
            prefill_first_ms=prefill_first_ms,
            prefill_steady_ms=prefill_steady_ms,
            decode_first_ms=decode_first_ms,
            decode_steady_ms=decode_steady_ms,
        )
        return {
            "candidate": candidate,
            "prefill_backend": prefill_backend,
            "decode_backend": decode_backend,
            "supported": True,
            "reason": "",
            "prefill_ms": prefill_ms,
            "prefill_first_ms": prefill_first_ms,
            "prefill_steady_ms": prefill_steady_ms,
            "decode_ms": decode_ms,
            "decode_first_ms": decode_first_ms,
            "decode_steady_ms": decode_steady_ms,
            "total_ms": total_ms,
            "weighted_total_ms": weighted,
            "online_conversion_ms": 0.0,
            "timing_mode": TIMING_MODE,
        }
    except Exception as exc:
        return {
            "candidate": candidate,
            "prefill_backend": "",
            "decode_backend": "",
            "supported": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "prefill_ms": "",
            "prefill_first_ms": "",
            "prefill_steady_ms": "",
            "decode_ms": "",
            "decode_first_ms": "",
            "decode_steady_ms": "",
            "total_ms": "",
            "weighted_total_ms": "",
            "online_conversion_ms": "",
            "timing_mode": TIMING_MODE,
        }
    finally:
        del base
        gc.collect()
        torch.cuda.empty_cache()


def manual_group_total_warm_e2e_ms(
    group: LinearGroup,
    scenario: ScenarioSpec,
    candidate: str,
    *,
    prefill_first_ms: float,
    prefill_steady_ms: float,
    decode_first_ms: float,
    decode_steady_ms: float,
) -> float:
    count = int(group.count)
    output_tokens = int(scenario.output_tokens)
    prefill_total = count * prefill_steady_ms
    if candidate == "dense_nvfp4_prefill_marlin_decode":
        per_module = prefill_steady_ms + decode_first_ms + max(output_tokens - 1, 0) * decode_steady_ms
        return count * per_module
    if output_tokens <= 0:
        return prefill_total
    decode_calls = count * output_tokens
    decode_total = decode_first_ms + max(decode_calls - 1, 0) * decode_steady_ms
    return prefill_total + decode_total


def pred_policy(
    model_key: str,
    scenario_name: str,
    groups: list[LinearGroup],
    scenario: ScenarioSpec,
    predictor: Any,
    out_dir: Path,
) -> HybridPolicy:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    decisions = []
    for group in groups:
        candidates = []
        for candidate in MANUAL_CANDIDATES:
            row = predict_candidate(group, scenario, candidate, predictor)
            row.update({"model": MODELS[model_key]["label"], "scenario": scenario_name, "linear_group": group.name, "n": group.n, "k": group.k, "count": group.count})
            rows.append(row)
            if row["supported"]:
                candidates.append(row)
        best = min(candidates, key=lambda row: float(row["weighted_total_ms"]))
        decisions.append(
            make_decision(
                group,
                selected_prefill=best["prefill_backend"],
                selected_decode=best["decode_backend"],
                total_ms=float(best["weighted_total_ms"]),
                prefill_ms=float(best["prefill_ms"]),
                decode_ms=float(best["decode_ms"]),
                conversion_ms=float(best["online_conversion_ms"]),
                candidates=[],
            )
        )
    write_csv(out_dir / f"{model_key}_pred_candidates.csv", rows)
    write_linear_summary(out_dir / f"{model_key}_linear_summary.csv", MODELS[model_key]["label"], scenario_name, decisions)
    return make_policy(scenario, decisions)


def predict_candidate(group: LinearGroup, scenario: ScenarioSpec, candidate: str, predictor: Any) -> dict[str, Any]:
    prefill_backend, decode_backend = candidate_backends(candidate)
    try:
        prefill_ms = predicted_latency(predictor, scenario.m_prefill, group.n, group.k, prefill_backend)
        decode_ms = 0.0 if scenario.output_tokens == 0 else predicted_latency(predictor, scenario.m_decode, group.n, group.k, decode_backend)
        conversion_ms = 0.0
        if candidate == "dense_nvfp4_prefill_marlin_decode":
            conversion_ms = sum_conversion_latency(predictor, group.n, group.k)
        total = prefill_ms + int(scenario.output_tokens) * decode_ms + conversion_ms
        return {
            "candidate": candidate,
            "prefill_backend": prefill_backend,
            "decode_backend": decode_backend,
            "supported": True,
            "reason": "",
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": total,
            "weighted_total_ms": group.count * total,
            "online_conversion_ms": conversion_ms,
            "timing_mode": TIMING_MODE,
        }
    except Exception as exc:
        return {
            "candidate": candidate,
            "prefill_backend": prefill_backend,
            "decode_backend": decode_backend,
            "supported": False,
            "reason": str(exc),
            "prefill_ms": "",
            "decode_ms": "",
            "total_ms": "",
            "weighted_total_ms": "",
            "online_conversion_ms": "",
            "timing_mode": TIMING_MODE,
        }


def predicted_latency(predictor: Any, m: int, n: int, k: int, backend: str) -> float:
    selection = predictor.predict(m, n, k)
    for record in selection.candidates:
        if record.kernel == backend:
            if not record.supported or record.latency_ms is None:
                raise ValueError(record.reason or f"unsupported:{backend}")
            return float(record.latency_ms)
    raise ValueError(f"missing_prediction:{backend}")


def sum_conversion_latency(predictor: Any, n: int, k: int) -> float:
    total = 0.0
    for record in predictor.predict_conversion(n, k):
        if not record.supported:
            raise ValueError(record.reason)
        total += float(record.latency_ms)
    return total


def make_base_linear(n: int, k: int, device: torch.device, seed: int) -> nn.Linear:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    linear = nn.Linear(k, n, bias=False, device=device, dtype=torch.bfloat16)
    linear.weight.data.normal_(mean=0.0, std=0.02, generator=generator)
    linear.eval()
    linear.requires_grad_(False)
    return linear


def make_candidate_module(candidate: str, base: nn.Linear, scenario: ScenarioSpec) -> tuple[nn.Module, str, str]:
    wrapper = load_wrapper()
    if candidate == "dense_bf16":
        return clone_linear(base), "dense_bf16", "dense_bf16"
    if candidate == "sparse_bf16":
        if (base.out_features, base.in_features) in SPARSE_BF16_BLOCKED_SHAPES:
            raise ValueError("sparse_bf16 blocked shape")
        sparse = wrapper.SparseBF16Linear.from_linear(base, prune=True).eval()
        return PaddedSparseBF16Linear(sparse, 8).eval(), "sparse_bf16", "sparse_bf16"
    if candidate == "dense_nvfp4":
        return wrapper.NVFP4Linear.from_linear(base, device=base.weight.device).eval(), "dense_nvfp4", "dense_nvfp4"
    if candidate == "sparse_nvfp4":
        sparse = wrapper.SparseNVFP4Linear.from_linear(base, device=base.weight.device, prune=True).eval()
        return PaddedSparseNVFP4Linear(sparse, 32).eval(), "sparse_nvfp4", "sparse_nvfp4"
    if candidate == "marlin_nvfp4":
        return wrapper.MarlinNVFP4Linear.from_linear(base, device=base.weight.device, activation_dtype=torch.bfloat16).eval(), "marlin_nvfp4", "marlin_nvfp4"
    if candidate == "dense_nvfp4_prefill_marlin_decode":
        canonical = wrapper.canonical_from_linear(base, device=base.weight.device)
        module = QwenHybridDenseNVFP4Linear(
            canonical,
            decode_activation_dtype=torch.bfloat16,
            marlin_m_threshold=scenario.m_decode,
            prefill_backend="dense_nvfp4",
            decode_backend="marlin_nvfp4",
        ).eval()
        return module, "dense_nvfp4", "marlin_nvfp4"
    raise ValueError(candidate)


def clone_linear(linear: nn.Linear) -> nn.Linear:
    out = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, device=linear.weight.device, dtype=torch.bfloat16)
    out.weight.data.copy_(linear.weight.detach().to(torch.bfloat16))
    if linear.bias is not None:
        out.bias.data.copy_(linear.bias.detach().to(torch.bfloat16))
    out.eval()
    out.requires_grad_(False)
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
        assert_finite(result)
        times.append(float(start.elapsed_time(end)))
        del result
    return sum(times) / len(times)


def assert_finite(tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all().item():
        raise RuntimeError("output contains NaN/Inf")


def candidate_backends(candidate: str) -> tuple[str, str]:
    if candidate == "dense_nvfp4_prefill_marlin_decode":
        return "dense_nvfp4", "marlin_nvfp4"
    return candidate, candidate


def make_policy(scenario: ScenarioSpec, decisions: list[LayerPolicyDecision]) -> HybridPolicy:
    return HybridPolicy(
        policy_format=POLICY_FORMAT,
        scenario={
            "batch_size": scenario.batch_size,
            "input_tokens": scenario.input_tokens,
            "output_tokens": scenario.output_tokens,
            "m_prefill": scenario.m_prefill,
            "m_decode": scenario.m_decode,
        },
        kernels=KERNELS,
        include_conversion_cost=True,
        modules=decisions,
    )


def make_decision(
    group: LinearGroup,
    *,
    selected_prefill: str,
    selected_decode: str,
    total_ms: float | None,
    prefill_ms: float | None,
    decode_ms: float | None,
    conversion_ms: float,
    candidates: list[StrategyCandidate],
) -> LayerPolicyDecision:
    return LayerPolicyDecision(
        name=group.name,
        n=group.n,
        k=group.k,
        count=group.count,
        selected_prefill_backend=selected_prefill,
        selected_decode_backend=selected_decode,
        selected_total_ms=total_ms,
        selected_prefill_ms=prefill_ms,
        selected_decode_ms=decode_ms,
        selected_conversion_ms=conversion_ms,
        strategy_candidates=candidates,
        prefill_candidates=[],
        decode_candidates=[],
        conversion_candidates=[],
        reason="",
    )


def write_policy_outputs(out_dir: Path, model_key: str, scenario_name: str, policy: HybridPolicy) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_policy_json(policy, out_dir / f"{model_key}_policy.json")
    write_policy_csv(policy, out_dir / f"{model_key}_policy.csv")
    write_leaf_readme(out_dir, model_key, scenario_name)


def single_linear_summary(
    args: argparse.Namespace,
    out_dir: Path,
    model_key: str,
    scenario_name: str,
    groups: list[LinearGroup],
    scenario: ScenarioSpec,
    method: str,
) -> None:
    rows = []
    total = 0.0
    for index, group in enumerate(groups):
        row = benchmark_manual_candidate(args, group, scenario, method, seed=2000 + index)
        used_fallback = False
        if not row["supported"] and method != "dense_bf16":
            failed_reason = row["reason"]
            row = benchmark_manual_candidate(args, group, scenario, "dense_bf16", seed=2000 + index)
            row["candidate"] = method
            row["fallback_backend"] = "dense_bf16"
            row["used_fallback"] = True
            row["fallback_reason"] = failed_reason
            used_fallback = True
        latency = float(row["weighted_total_ms"]) if row["supported"] else 0.0
        total += latency
        row.update(
            {
                "model": MODELS[model_key]["label"],
                "scenario": scenario_name,
                "linear_group": group.name,
                "n": group.n,
                "k": group.k,
                "count": group.count,
                "used_fallback": used_fallback,
                "timing_mode": TIMING_MODE,
            }
        )
        rows.append(row)
    rows.append({"model": MODELS[model_key]["label"], "scenario": scenario_name, "linear_group": "__TOTAL__", "candidate": method, "weighted_total_ms": total, "timing_mode": TIMING_MODE})
    write_csv(out_dir / f"{model_key}_linear_summary.csv", rows)


def write_linear_summary(path: Path, model: str, scenario: str, decisions: list[LayerPolicyDecision]) -> None:
    rows = []
    total = 0.0
    for decision in decisions:
        weighted = decision.selected_total_ms or 0.0
        total += weighted
        rows.append(
            {
                "model": model,
                "scenario": scenario,
                "linear_group": decision.name,
                "n": decision.n,
                "k": decision.k,
                "count": decision.count,
                "prefill_backend": decision.selected_prefill_backend,
                "decode_backend": decision.selected_decode_backend,
                "prefill_ms": decision.selected_prefill_ms,
                "decode_ms": decision.selected_decode_ms,
                "conversion_ms": decision.selected_conversion_ms,
                "weighted_total_ms": weighted,
                "timing_mode": TIMING_MODE,
            }
        )
    rows.append({"model": model, "scenario": scenario, "linear_group": "__TOTAL__", "weighted_total_ms": total, "timing_mode": TIMING_MODE})
    write_csv(path, rows)


def run_full_e2e(
    args: argparse.Namespace,
    model_key: str,
    scenario_name: str,
    out_dir: Path,
    *,
    method: str,
    policy_path: Path,
    dtype: torch.dtype,
) -> None:
    output_csv = out_dir / f"{model_key}_full_e2e.csv"
    if args.skip_existing_e2e and output_csv.exists():
        return
    model = load_model(model_key, dtype=dtype, gpu=args.gpu)
    report = None
    policy_method = "dense_bf16"
    if "single" == method:
        policy = json.loads(policy_path.read_text())
        policy_method = policy["modules"][0]["selected_prefill_backend"]
    if method in {"manual", "pred"} or policy_method != "dense_bf16":
        report = apply_policy(model_key, model, policy_path, dtype)
    scenario = SCENARIOS[scenario_name]
    result = benchmark_model(model, scenario, args.gpu, args.warmup_iters)
    row = {
        "model": MODELS[model_key]["label"],
        "scenario": scenario_name,
        "method_family": method,
        "policy_or_method": policy_method if method == "single" else method,
        "batch_size": scenario["batch_size"],
        "input_tokens": scenario["input_tokens"],
        "output_tokens": scenario["output_tokens"],
        "timing_mode": TIMING_MODE,
        "prefill_ms": result["prefill_ms"],
        "decode_avg_ms": result["decode_avg_ms"],
        "decode_first_ms": result["decode_first_ms"],
        "decode_steady_ms": result["decode_steady_ms"],
        "decode_x_n_ms": scenario["output_tokens"] * result["decode_avg_ms"],
        "e2e_ms": result["prefill_ms"] + scenario["output_tokens"] * result["decode_avg_ms"],
        "replaced_linear_count": "" if report is None else getattr(report, "replaced_linear_count", ""),
        "skipped_linear_count": "" if report is None else getattr(report, "skipped_linear_count", ""),
        "backend_counts": "" if report is None else dict(getattr(report, "backend_counts", {})),
        "policy_json": str(policy_path),
    }
    write_csv(output_csv, [row])
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_key: str, *, dtype: torch.dtype, gpu: int) -> nn.Module:
    if MODELS[model_key]["family"] == "qwen3_5":
        from fake.models.qwen3_5 import load_qwen3_5_dense
        model, _ = load_qwen3_5_dense(MODELS[model_key]["path"], device=f"cuda:{gpu}", torch_dtype=dtype)
        return model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_key]["path"], torch_dtype=dtype, local_files_only=True)
    model = model.to(f"cuda:{gpu}")
    model.eval()
    model.requires_grad_(False)
    return model


def apply_policy(model_key: str, model: nn.Module, policy_path: Path, dtype: torch.dtype) -> Any:
    if MODELS[model_key]["family"] == "qwen3_5":
        from fake.models.qwen3_5_kernels import replace_linear_with_qwen_predictor_hybrid
        return replace_linear_with_qwen_predictor_hybrid(model, policy_path=policy_path, activation_dtype=dtype)
    from fake.models.llama_kernels import replace_linear_with_llama_predictor_hybrid
    return replace_linear_with_llama_predictor_hybrid(model, policy_path=policy_path, activation_dtype=dtype)


@torch.inference_mode()
def benchmark_model(model: nn.Module, scenario: dict[str, int], gpu: int, warmup_iters: int) -> dict[str, float]:
    device = f"cuda:{gpu}"
    for _ in range(warmup_iters):
        ids = torch.randint(0, 1000, (scenario["batch_size"], min(32, scenario["input_tokens"])), device=device)
        _ = model(ids, use_cache=False)
    torch.cuda.synchronize()
    input_ids = torch.randint(0, 1000, (scenario["batch_size"], scenario["input_tokens"]), device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = model(input_ids, use_cache=scenario["output_tokens"] > 0)
    end.record()
    torch.cuda.synchronize()
    prefill_ms = float(start.elapsed_time(end))
    if scenario["output_tokens"] == 0:
        return {"prefill_ms": prefill_ms, "decode_avg_ms": 0.0, "decode_first_ms": 0.0, "decode_steady_ms": 0.0}
    past_key_values = out.past_key_values
    next_token = torch.randint(0, 1000, (scenario["batch_size"], 1), device=device)
    times = []
    for _ in range(scenario["output_tokens"]):
        start.record()
        out = model(next_token, past_key_values=past_key_values, use_cache=True)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
        past_key_values = out.past_key_values
        next_token = torch.randint(0, 1000, (scenario["batch_size"], 1), device=device)
    return {
        "prefill_ms": prefill_ms,
        "decode_avg_ms": sum(times) / len(times),
        "decode_first_ms": times[0],
        "decode_steady_ms": sum(times[2:]) / max(len(times[2:]), 1),
    }


def load_wrapper() -> Any:
    import importlib
    for module_name in ("fake.kernels.cutlass.cutlass_wrapper.cutlass_wrapper", "cutlass_wrapper"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass
    raise RuntimeError("CUTLASS wrapper package is not importable")


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


def write_leaf_readme(out_dir: Path, model_key: str, scenario_name: str) -> None:
    scenario = SCENARIOS[scenario_name]
    (out_dir / "README.md").write_text(
        f"# {scenario_name} results\n\n"
        f"- Model included in files: `{model_key}`\n"
        f"- Scenario: `batch_size={scenario['batch_size']},input_tokens={scenario['input_tokens']},output_tokens={scenario['output_tokens']}`\n"
        f"- Timing mode: `{TIMING_MODE}`\n"
        "- `*_linear_summary.csv` contains warm-E2E-aligned linear-module aggregate latency.\n"
        "- `*_full_e2e.csv` contains real warmed full-model E2E latency when `--run-e2e` was used.\n"
    )


def write_root_readme(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# Main Hybrid Policy Retest\n\n"
        "This directory contains warm-E2E-aligned single-backend, manual-policy, and predictor-policy retests for "
        "Llama-2-7B, Llama-3.1-8B, and Qwen3.5-9B under `prefill_only`, `normal_01`, and `normal_02`.\n\n"
        "The final metric is warmed full-model E2E latency. Linear-module aggregate latency is aligned to the same warm prefill semantics.\n"
        "See `ANALYSIS.md` for the E2E ranking and main observations.\n"
    )


def write_comparison(root: Path) -> None:
    comp = root / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    e2e_rows = []
    linear_rows = []
    for path in root.rglob("*_full_e2e.csv"):
        with path.open(newline="") as f:
            e2e_rows.extend(csv.DictReader(f))
    for path in root.rglob("*_linear_summary.csv"):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("linear_group") == "__TOTAL__":
                    row["source_file"] = str(path.relative_to(root))
                    linear_rows.append(row)
    write_csv(comp / "full_e2e_summary.csv", e2e_rows)
    write_csv(comp / "linear_latency_summary.csv", linear_rows)
    write_policy_diff(root, comp)
    (comp / "README.md").write_text(
        "# Comparison\n\n"
        "- `full_e2e_summary.csv`: real full-model E2E rows.\n"
        "- `linear_latency_summary.csv`: linear aggregate totals.\n"
        "- `manual_vs_pred_policy_diff.csv`: policy differences between manual and prediction.\n"
    )


def write_policy_diff(root: Path, comp: Path) -> None:
    rows = []
    for scenario in SCENARIOS:
        for model_key in MODELS:
            manual = root / "manual" / scenario / f"{model_key}_policy.json"
            pred = root / "pred" / scenario / f"{model_key}_policy.json"
            if not manual.exists() or not pred.exists():
                continue
            m_payload = json.loads(manual.read_text())
            p_payload = json.loads(pred.read_text())
            pred_by_name = {row["name"]: row for row in p_payload["modules"]}
            for m_row in m_payload["modules"]:
                p_row = pred_by_name.get(m_row["name"])
                if p_row is None:
                    continue
                rows.append(
                    {
                        "model": MODELS[model_key]["label"],
                        "scenario": scenario,
                        "linear_group": m_row["name"],
                        "count": m_row["count"],
                        "manual": f"{m_row.get('selected_prefill_backend')}->{m_row.get('selected_decode_backend')}",
                        "pred": f"{p_row.get('selected_prefill_backend')}->{p_row.get('selected_decode_backend')}",
                        "same": (
                            m_row.get("selected_prefill_backend") == p_row.get("selected_prefill_backend")
                            and m_row.get("selected_decode_backend") == p_row.get("selected_decode_backend")
                        ),
                    }
                )
    write_csv(comp / "manual_vs_pred_policy_diff.csv", rows)


if __name__ == "__main__":
    main()
