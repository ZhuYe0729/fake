#!/usr/bin/env python3
"""Benchmark offline vLLM generate for phase-switch and single-method models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--trace-json")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--cutlass-wrapper-path", required=True)
    parser.add_argument("--phase-switch", action="store_true")
    return parser.parse_args()


def make_prompt(tokenizer, input_len: int, idx: int) -> str:
    seed = (
        f"Request {idx}. This benchmark prompt is intentionally repetitive "
        "so that tokenization is stable across runs. "
    )
    ids = tokenizer.encode(seed, add_special_tokens=False)
    if not ids:
        raise RuntimeError("tokenizer produced an empty prompt")
    repeated = (ids * ((input_len // len(ids)) + 2))[:input_len]
    return tokenizer.decode(repeated, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    artifact_dir = Path(args.artifact_dir)
    sys.path.insert(0, str(repo_root / "vllm"))
    sys.path.insert(0, str(artifact_dir))
    sys.path.insert(0, args.cutlass_wrapper_path)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["PHASE_SWITCH_MYTEST_CUTLASS_WRAPPER_PATH"] = (
        args.cutlass_wrapper_path)
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = args.cutlass_wrapper_path
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = (
        args.cutlass_wrapper_path)

    phase_switch_mytest = None
    if args.phase_switch:
        import phase_switch_mytest as _phase_switch_mytest
        phase_switch_mytest = _phase_switch_mytest

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    prompts = [
        make_prompt(tokenizer, args.input_len, i)
        for i in range(args.batch_size)
    ]
    prompt_token_counts = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]

    init_start = time.perf_counter()
    llm = LLM(model=args.model,
              dtype="bfloat16",
              trust_remote_code=True,
              enforce_eager=True,
              skip_mm_profiling=True,
              enable_chunked_prefill=False,
              max_num_batched_tokens=args.max_num_batched_tokens)
    init_s = time.perf_counter() - init_start

    if phase_switch_mytest is not None:
        phase_switch_mytest.enable_phase_switch()

    start = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=args.output_len, temperature=0.0))
    elapsed_s = time.perf_counter() - start

    output_token_counts = [
        len(output.outputs[0].token_ids)
        for output in outputs
    ]
    if any(count == 0 for count in output_token_counts):
        raise AssertionError("benchmark produced an empty output")

    total_prompt_tokens = sum(prompt_token_counts)
    total_output_tokens = sum(output_token_counts)
    row = {
        "label": args.label,
        "scenario": args.scenario,
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "batch_size": args.batch_size,
        "init_s": init_s,
        "generate_s": elapsed_s,
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_prompt_tokens + total_output_tokens,
        "output_tokens_per_s": total_output_tokens / elapsed_s,
        "total_tokens_per_s": (total_prompt_tokens + total_output_tokens) /
        elapsed_s,
        "avg_latency_s_per_request": elapsed_s / args.batch_size,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(row, f, indent=2)
        f.write("\n")

    if phase_switch_mytest is not None and args.trace_json:
        phase_switch_mytest.dump_trace(args.trace_json)


if __name__ == "__main__":
    main()
