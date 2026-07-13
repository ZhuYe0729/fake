#!/usr/bin/env python3
"""Benchmark Llama2 vLLM prefill baseline models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BASELINE_ROOT.parents[3]
DEFAULT_DENSE_BF16_MODEL = Path("/root/wja/data/models/LLM-Research/llama-2-7b")
MODEL_SPECS = {
    "dense_bf16": DEFAULT_DENSE_BF16_MODEL,
    "dense_nvfp4": BASELINE_ROOT / "uniform_dense_nvfp4",
    "sparse_bf16": BASELINE_ROOT / "uniform_sparse_bf16",
    "sparse_nvfp4": BASELINE_ROOT / "uniform_sparse_nvfp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT / "benchmarks/prefill_vllm")
    parser.add_argument("--methods", default="dense_bf16,dense_nvfp4,sparse_bf16,sparse_nvfp4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--no-enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for method in methods:
        model_path = MODEL_SPECS[method]
        if not model_path.exists():
            raise FileNotFoundError(f"{method} model path does not exist: {model_path}")
        summary, iterations = benchmark_method(method, model_path, args)
        summaries.append(summary)
        write_csv(args.output_dir / f"{method}_iterations.csv", iterations)
        write_json(args.output_dir / f"{method}_summary.json", summary)

    dense_ms = next(row["median_ms"] for row in summaries if row["method"] == "dense_bf16")
    for row in summaries:
        row["speedup_vs_dense_bf16"] = dense_ms / row["median_ms"]
    write_csv(args.output_dir / "prefill_vllm_summary.csv", summaries)
    write_json(
        args.output_dir / "prefill_vllm_metadata.json",
        {
            "scenario": "prefill_plus_1_decode",
            "note": "vLLM generate uses max_tokens=1; this is a prefill-only approximation with one decode token.",
            "batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "warmup_iters": args.warmup_iters,
            "iters": args.iters,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": not args.no_enforce_eager,
            "device": args.device,
            "methods": methods,
        },
    )


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [method for method in methods if method not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={sorted(MODEL_SPECS)}")
    return methods


def benchmark_method(
    method: str, model_path: Path, args: argparse.Namespace
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"loading {method}: {model_path}", flush=True)
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = int(getattr(tokenizer, "vocab_size", 32000))
    prompts = make_prompts(
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        vocab_size=vocab_size,
        seed=args.seed,
    )
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )

    for i in range(args.warmup_iters):
        run_once(llm, prompts, sampling)
        print(f"{method} warmup {i + 1}/{args.warmup_iters}", flush=True)

    rows: list[dict[str, Any]] = []
    for i in range(args.iters):
        elapsed = run_once(llm, prompts, sampling)
        row = {
            "method": method,
            "iter": i,
            "elapsed_ms": elapsed * 1000.0,
            "batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
        }
        rows.append(row)
        print(f"{method} iter {i + 1}/{args.iters}: {row['elapsed_ms']:.3f} ms", flush=True)

    del llm
    torch.cuda.empty_cache()
    times = [float(row["elapsed_ms"]) for row in rows]
    total_prompt_tokens = args.batch_size * args.prompt_len
    median_ms = statistics.median(times)
    mean_ms = statistics.mean(times)
    summary = {
        "method": method,
        "model_path": str(model_path),
        "scenario": "prefill_plus_1_decode",
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "iters": args.iters,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "prompt_tokens_per_s_median": total_prompt_tokens / (median_ms / 1000.0),
        "prompt_tokens_per_s_mean": total_prompt_tokens / (mean_ms / 1000.0),
        "speedup_vs_dense_bf16": math.nan,
    }
    return summary, rows


def make_prompts(
    *, batch_size: int, prompt_len: int, vocab_size: int, seed: int
) -> list[TokensPrompt]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    low = 100
    high = max(low + 1, vocab_size - 1)
    prompts = []
    for _ in range(batch_size):
        ids = torch.randint(low=low, high=high, size=(prompt_len,), generator=generator).tolist()
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    return prompts


def run_once(
    llm: LLM, prompts: list[TokensPrompt], sampling: SamplingParams
) -> float:
    start = time.perf_counter()
    llm.generate(prompts, sampling, use_tqdm=False)
    return time.perf_counter() - start


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
