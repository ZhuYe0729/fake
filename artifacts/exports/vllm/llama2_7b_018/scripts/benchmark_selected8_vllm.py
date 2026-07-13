#!/usr/bin/env python3
"""Benchmark selected 8 Llama2 scenarios with vLLM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_DENSE_BF16_MODEL = Path("/root/wja/data/models/LLM-Research/llama-2-7b")


@dataclass(frozen=True)
class Scenario:
    name: str
    batch_size: int
    input_len: int
    output_tokens: int

    @property
    def max_model_len(self) -> int:
        return self.input_len + self.output_tokens


SCENARIOS = (
    Scenario("single_long_prefill_short_decode", 1, 8192, 16),
    Scenario("small_batch_long_prefill", 2, 4096, 16),
    Scenario("b4_medium_prefill", 4, 2048, 16),
    Scenario("b4_long_prefill", 4, 4096, 32),
    Scenario("b8_mixed_long_prefill", 8, 2048, 64),
    Scenario("b16_mixed", 16, 1024, 64),
    Scenario("b32_throughput_mixed", 32, 512, 64),
    Scenario("b64_high_batch", 64, 256, 128),
)

MODEL_SPECS = {
    "dense_bf16": DEFAULT_DENSE_BF16_MODEL,
    "dense_nvfp4": BASELINE_ROOT / "uniform_dense_nvfp4",
    "sparse_bf16": BASELINE_ROOT / "uniform_sparse_bf16",
    "sparse_nvfp4": BASELINE_ROOT / "uniform_sparse_nvfp4",
    "marlin_nvfp4": BASELINE_ROOT / "uniform_marlin_nvfp4",
}
HETERO_SCENARIO_MODEL = {
    "single_long_prefill_short_decode": BASELINE_ROOT / "hetero_strategy_a",
    "small_batch_long_prefill": BASELINE_ROOT / "hetero_strategy_a",
    "b4_medium_prefill": BASELINE_ROOT / "hetero_strategy_b",
    "b4_long_prefill": BASELINE_ROOT / "hetero_strategy_b",
    "b8_mixed_long_prefill": BASELINE_ROOT / "hetero_strategy_c",
    "b16_mixed": BASELINE_ROOT / "hetero_strategy_c",
    "b32_throughput_mixed": BASELINE_ROOT / "hetero_strategy_c",
    "b64_high_batch": BASELINE_ROOT / "hetero_strategy_c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASELINE_ROOT / "benchmarks/selected_8_scenarios_vllm",
    )
    parser.add_argument(
        "--methods",
        default="dense_bf16,dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4,hetero",
    )
    parser.add_argument("--scenarios", default=",".join(s.name for s in SCENARIOS))
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--no-hf-max-position-override",
        action="store_true",
        help="Do not override max_position_embeddings for synthetic long-context speed tests.",
    )
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    scenarios = parse_scenarios(args.scenarios)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    all_iterations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for method in methods:
        groups = model_groups_for_method(method, scenarios)
        for model_path, group_scenarios in groups.items():
            try:
                summaries, iterations = benchmark_model_group(
                    method=method,
                    model_path=model_path,
                    scenarios=group_scenarios,
                    args=args,
                )
                all_summaries.extend(summaries)
                all_iterations.extend(iterations)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                errors.append(
                    {
                        "method": method,
                        "model_path": str(model_path),
                        "scenarios": [s.name for s in group_scenarios],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"ERROR {method} {model_path}: {type(exc).__name__}: {exc}", flush=True)

    add_speedups(all_summaries)
    write_csv(args.output_dir / "selected8_vllm_summary.csv", all_summaries)
    write_csv(args.output_dir / "selected8_vllm_iterations.csv", all_iterations)
    write_json(
        args.output_dir / "selected8_vllm_metadata.json",
        {
            "scenarios": [asdict(s) for s in scenarios],
            "methods": methods,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": not args.no_enforce_eager,
            "device": args.device,
            "errors": errors,
        },
    )


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    supported = set(MODEL_SPECS) | {"hetero"}
    unknown = [method for method in methods if method not in supported]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={sorted(supported)}")
    return methods


def parse_scenarios(spec: str) -> list[Scenario]:
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    by_name = {scenario.name: scenario for scenario in SCENARIOS}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}; supported={sorted(by_name)}")
    return [by_name[name] for name in requested]


def model_groups_for_method(method: str, scenarios: list[Scenario]) -> dict[Path, list[Scenario]]:
    if method != "hetero":
        path = MODEL_SPECS[method]
        if not path.exists():
            raise FileNotFoundError(f"{method} model path does not exist: {path}")
        return {path: scenarios}

    groups: dict[Path, list[Scenario]] = {}
    for scenario in scenarios:
        path = HETERO_SCENARIO_MODEL[scenario.name]
        if not path.exists():
            raise FileNotFoundError(f"hetero model path does not exist: {path}")
        groups.setdefault(path, []).append(scenario)
    return groups


def benchmark_model_group(
    *,
    method: str,
    model_path: Path,
    scenarios: list[Scenario],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_model_len = args.max_model_len or max(s.max_model_len for s in scenarios)
    print(f"loading {method}: {model_path} max_model_len={max_model_len}", flush=True)
    hf_overrides = long_context_hf_overrides(model_path, max_model_len, args)
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        enable_prefix_caching=False,
        hf_overrides=hf_overrides,
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000))
    summaries: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []

    for scenario in scenarios:
        prompts = make_prompts(
            batch_size=scenario.batch_size,
            prompt_len=scenario.input_len,
            vocab_size=vocab_size,
            seed=args.seed,
        )
        sampling = SamplingParams(
            max_tokens=scenario.output_tokens,
            min_tokens=scenario.output_tokens,
            temperature=0.0,
            ignore_eos=True,
            detokenize=False,
        )
        for i in range(args.warmup_iters):
            run_once(llm, prompts, sampling)
            print(f"{method} {scenario.name} warmup {i + 1}/{args.warmup_iters}", flush=True)

        rows: list[dict[str, Any]] = []
        for i in range(args.iters):
            elapsed = run_once(llm, prompts, sampling)
            row = {
                "method": method,
                "scenario": scenario.name,
                "iter": i,
                "elapsed_ms": elapsed * 1000.0,
                "batch_size": scenario.batch_size,
                "input_len": scenario.input_len,
                "output_tokens": scenario.output_tokens,
                "model_path": str(model_path),
            }
            rows.append(row)
            iterations.append(row)
            print(
                f"{method} {scenario.name} iter {i + 1}/{args.iters}: "
                f"{row['elapsed_ms']:.3f} ms",
                flush=True,
            )
        summaries.append(summarize_rows(method, model_path, scenario, rows))

    del llm
    torch.cuda.empty_cache()
    return summaries, iterations


def long_context_hf_overrides(
    model_path: Path, max_model_len: int, args: argparse.Namespace
) -> dict[str, Any] | None:
    if args.no_hf_max_position_override:
        return None
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
    configured = int(getattr(config, "max_position_embeddings", 0) or 0)
    if configured and max_model_len > configured:
        return {"max_position_embeddings": max_model_len}
    return None


def make_prompts(
    *, batch_size: int, prompt_len: int, vocab_size: int, seed: int
) -> list[TokensPrompt]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    low = 100
    high = max(low + 1, vocab_size - 1)
    return [
        TokensPrompt(
            prompt_token_ids=torch.randint(
                low=low, high=high, size=(prompt_len,), generator=generator
            ).tolist()
        )
        for _ in range(batch_size)
    ]


def run_once(llm: LLM, prompts: list[TokensPrompt], sampling: SamplingParams) -> float:
    start = time.perf_counter()
    llm.generate(prompts, sampling, use_tqdm=False)
    return time.perf_counter() - start


def summarize_rows(
    method: str, model_path: Path, scenario: Scenario, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    times = [float(row["elapsed_ms"]) for row in rows]
    median_ms = statistics.median(times)
    mean_ms = statistics.mean(times)
    total_input_tokens = scenario.batch_size * scenario.input_len
    total_output_tokens = scenario.batch_size * scenario.output_tokens
    return {
        "method": method,
        "scenario": scenario.name,
        "model_path": str(model_path),
        "batch_size": scenario.batch_size,
        "input_len": scenario.input_len,
        "output_tokens": scenario.output_tokens,
        "iters": len(rows),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "input_tokens_per_s_median": total_input_tokens / (median_ms / 1000.0),
        "total_tokens_per_s_median": (total_input_tokens + total_output_tokens)
        / (median_ms / 1000.0),
        "speedup_vs_dense_bf16": math.nan,
    }


def add_speedups(rows: list[dict[str, Any]]) -> None:
    dense_by_scenario = {
        row["scenario"]: float(row["median_ms"])
        for row in rows
        if row["method"] == "dense_bf16"
    }
    for row in rows:
        dense_ms = dense_by_scenario.get(row["scenario"])
        if dense_ms:
            row["speedup_vs_dense_bf16"] = dense_ms / float(row["median_ms"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
