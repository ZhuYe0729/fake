#!/usr/bin/env python3
"""Benchmark Llama-3.1-8B-Instruct baselines on the selected vLLM scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent
MODEL_PATH = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")

METHOD_PATHS = {
    "dense_bf16": MODEL_PATH,
    "dense_nvfp4": BASELINE_ROOT / "checkpoints/uniform_dense_nvfp4",
    "sparse_bf16": BASELINE_ROOT / "checkpoints/uniform_sparse_bf16",
    "sparse_nvfp4": BASELINE_ROOT / "checkpoints/uniform_sparse_nvfp4",
    "marlin_nvfp4": BASELINE_ROOT / "checkpoints/uniform_marlin_nvfp4",
}
METHODS = tuple(METHOD_PATHS)


@dataclass(frozen=True)
class Scenario:
    name: str
    batch: int
    input_seq: int
    output_seq: int

    @property
    def max_model_len(self) -> int:
        return self.input_seq + self.output_seq

    @property
    def prompt_tokens(self) -> int:
        return self.batch * self.input_seq

    @property
    def output_tokens(self) -> int:
        return self.batch * self.output_seq


SCENARIOS = (
    Scenario("prefill_only", 8, 2048, 1),
    Scenario("prefill_decode", 16, 2048, 80),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--scenarios", default=",".join(s.name for s in SCENARIOS))
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT / "results/speed")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--no-hf-max-position-override", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.pop("_CUDA_COMPAT_STATUS", None)
    args = parse_args()
    methods = parse_methods(args.methods)
    scenarios = parse_scenarios(args.scenarios)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []

    for method in methods:
        method_rows, method_iterations = benchmark_method(method, scenarios, args)
        rows.extend(method_rows)
        iterations.extend(method_iterations)
        write_csv(args.output_dir / "summary.csv", rows)
        write_csv(args.output_dir / "iterations.csv", iterations)

    metadata = {
        "methods": methods,
        "scenarios": [s.__dict__ for s in scenarios],
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": not args.no_enforce_eager,
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "ttft_tpot_note": "TTFT is measured with output_seq=1. TPOT for output_seq>1 is (e2e_ms - ttft_ms)/(output_seq - 1).",
    }
    write_json(args.output_dir / "metadata.json", metadata)


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHOD_PATHS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={list(METHOD_PATHS)}")
    return methods


def parse_scenarios(spec: str) -> list[Scenario]:
    by_name = {scenario.name: scenario for scenario in SCENARIOS}
    names = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}; supported={list(by_name)}")
    return [by_name[name] for name in names]


def benchmark_method(
    method: str, scenarios: list[Scenario], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    model_path = METHOD_PATHS[method]

    for scenario in scenarios:
        llm = None
        try:
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            print(
                f"loading method={method} scenario={scenario.name} model={model_path}",
                flush=True,
            )
            llm = load_llm(model_path, scenario, args)
            tokenizer = llm.get_tokenizer()
            vocab_size = int(getattr(tokenizer, "vocab_size", 32000))

            ttft_summary, ttft_iters = run_timed_generate(
                llm=llm,
                method=method,
                scenario=scenario,
                output_seq=1,
                vocab_size=vocab_size,
                args=args,
                phase="ttft",
            )
            final_summary, final_iters = run_timed_generate(
                llm=llm,
                method=method,
                scenario=scenario,
                output_seq=scenario.output_seq,
                vocab_size=vocab_size,
                args=args,
                phase="main",
            )
            row = make_summary_row(method, model_path, scenario, ttft_summary, final_summary)
            rows.append(row)
            iterations.extend(ttft_iters)
            iterations.extend(final_iters)
        except Exception as exc:
            error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            print(f"FAIL method={method} scenario={scenario.name}: {exc}", flush=True)
            rows.append(status_row(method, model_path, scenario, "FAIL", error))
            if not args.continue_on_error:
                raise
        finally:
            cleanup_llm(llm)
    return rows, iterations


def load_llm(model_path: Path, scenario: Scenario, args: argparse.Namespace) -> LLM:
    return LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=scenario.max_model_len,
        max_num_seqs=scenario.batch,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        enable_prefix_caching=False,
        hf_overrides=long_context_hf_overrides(model_path, scenario.max_model_len, args),
    )


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


def run_timed_generate(
    *,
    llm: LLM,
    method: str,
    scenario: Scenario,
    output_seq: int,
    vocab_size: int,
    args: argparse.Namespace,
    phase: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    prompts = make_prompts(scenario.batch, scenario.input_seq, vocab_size, args.seed)
    sampling = SamplingParams(
        max_tokens=output_seq,
        min_tokens=output_seq,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    elapsed_ms: list[float] = []
    iter_rows: list[dict[str, Any]] = []
    total = args.warmup_iters + args.iters
    for idx in range(total):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        generated = sum(len(output.outputs[0].token_ids) for output in outputs)
        is_warmup = idx < args.warmup_iters
        row = {
            "method": method,
            "scenario": scenario.name,
            "phase": phase,
            "iteration": idx,
            "warmup": is_warmup,
            "batch": scenario.batch,
            "input_seq": scenario.input_seq,
            "output_seq": output_seq,
            "elapsed_ms": elapsed,
            "generated_tokens": generated,
        }
        iter_rows.append(row)
        if not is_warmup:
            elapsed_ms.append(elapsed)
    return stats(elapsed_ms), iter_rows


def make_prompts(batch: int, input_seq: int, vocab_size: int, seed: int) -> list[TokensPrompt]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + batch * 1000003 + input_seq)
    low = 100
    high = max(low + 1, min(vocab_size, 30000))
    token_ids = torch.randint(low, high, (batch, input_seq), generator=generator, dtype=torch.int64)
    return [TokensPrompt(prompt_token_ids=row.tolist()) for row in token_ids]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("no timed iterations")
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def make_summary_row(
    method: str,
    model_path: Path,
    scenario: Scenario,
    ttft: dict[str, float],
    final: dict[str, float],
) -> dict[str, Any]:
    e2e_ms = final["median_ms"]
    ttft_ms = ttft["median_ms"]
    if scenario.output_seq > 1:
        tpot_ms = max(e2e_ms - ttft_ms, 0.0) / (scenario.output_seq - 1)
    else:
        tpot_ms = 0.0
    elapsed_s = e2e_ms / 1000.0
    return {
        "status": "OK",
        "method": method,
        "model_path": str(model_path),
        "scenario": scenario.name,
        "batch": scenario.batch,
        "input_seq": scenario.input_seq,
        "output_seq": scenario.output_seq,
        "e2e_median_ms": e2e_ms,
        "e2e_mean_ms": final["mean_ms"],
        "e2e_min_ms": final["min_ms"],
        "e2e_max_ms": final["max_ms"],
        "e2e_std_ms": final["std_ms"],
        "ttft_median_ms": ttft_ms,
        "ttft_mean_ms": ttft["mean_ms"],
        "tpot_ms": tpot_ms,
        "requests_per_s": scenario.batch / elapsed_s,
        "prompt_tokens_per_s": scenario.prompt_tokens / elapsed_s,
        "output_tokens_per_s": scenario.output_tokens / elapsed_s,
        "total_tokens_per_s": (scenario.prompt_tokens + scenario.output_tokens) / elapsed_s,
    }


def status_row(
    method: str, model_path: Path, scenario: Scenario, status: str, error: str
) -> dict[str, Any]:
    return {
        "status": status,
        "method": method,
        "model_path": str(model_path),
        "scenario": scenario.name,
        "batch": scenario.batch,
        "input_seq": scenario.input_seq,
        "output_seq": scenario.output_seq,
        "error": error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cleanup_llm(llm: LLM | None) -> None:
    del llm
    cleanup_cuda()


def cleanup_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
