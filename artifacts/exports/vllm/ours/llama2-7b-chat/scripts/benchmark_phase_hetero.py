#!/usr/bin/env python3
"""Benchmark one-shot vLLM phase-heterogeneous checkpoints like the baseline."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS_ROOT = Path("/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper")
SCENARIOS = {"prefill_only": (8, 2048, 1), "prefill_decode": (16, 2048, 80)}


@dataclass(frozen=True)
class Scenario:
    name: str; batch: int; input_seq: int; output_seq: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--batch", type=int,
                        help="Override the scenario batch size for a diagnostic run.")
    parser.add_argument("--input-seq", type=int,
                        help="Override the scenario input length for a diagnostic run.")
    parser.add_argument("--output-seq", type=int,
                        help="Override the scenario output length for a diagnostic run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, default=VLLM_ROOT)
    parser.add_argument("--cutlass-wrapper-path", type=Path, default=CUTLASS_ROOT)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-memory-bytes", type=int,
                        help="Fix V1 KV-cache capacity across policies.")
    parser.add_argument("--kv-cache-dtype",
                        help="Optional vLLM KV-cache dtype, e.g. fp8.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single-phase", choices=("ttft", "main"))
    parser.add_argument("--single-output", type=Path)
    parser.add_argument("--single-samples-dir", type=Path,
                        help="Optional directory for per-sample JSON files in single mode.")
    parser.add_argument("--reuse-llm", action="store_true",
                        help="Run repeated phase-E2E samples in one LLM instance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_batch, default_input, default_output = SCENARIOS[args.scenario]
    scenario = Scenario(args.scenario, args.batch or default_batch,
                        args.input_seq or default_input,
                        args.output_seq or default_output)
    if args.single_phase:
        run_single_process(args, scenario)
        return
    output_dir = args.output_dir; output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_llm:
        ttft, ttft_rows, main_stats, main_rows = run_reuse_benchmark(args, scenario)
        execution = "one_vllm_process_one_llm"
    else:
        ttft, ttft_rows = run_isolated_phase(args, scenario, 1, "ttft")
        main_stats, main_rows = run_isolated_phase(args, scenario, scenario.output_seq, "main")
        execution = "one_vllm_process_per_sample"
    tpot = 0.0 if scenario.output_seq <= 1 else (main_stats["median_ms"] - ttft["median_ms"]) / (scenario.output_seq - 1)
    summary = {"method": "ours_max_speed", "scenario": scenario.name, "checkpoint": str(args.checkpoint), "status": "OK",
               "batch": scenario.batch, "input_seq": scenario.input_seq, "output_seq": scenario.output_seq,
               "e2e_mean_ms": main_stats["mean_ms"], "e2e_median_ms": main_stats["median_ms"], "ttft_median_ms": ttft["median_ms"],
               "tpot_ms": tpot, "total_tokens_per_s": scenario.batch * (scenario.input_seq + scenario.output_seq) * 1000.0 / main_stats["mean_ms"]}
    write_csv(output_dir / "iterations.csv", ttft_rows + main_rows)
    write_csv(output_dir / "summary.csv", [summary])
<<<<<<< Updated upstream
    (output_dir / "metadata.json").write_text(json.dumps({"scenario": scenario.__dict__, "warmup_iters": args.warmup_iters, "iters": args.iters, "phase_runtime": "phase_hetero_mytest", "execution": "one_vllm_process_per_sample", "kv_cache_memory_bytes": args.kv_cache_memory_bytes, "kv_cache_dtype": args.kv_cache_dtype}, indent=2) + "\n")
=======
    (output_dir / "metadata.json").write_text(json.dumps({"scenario": scenario.__dict__, "warmup_iters": args.warmup_iters, "iters": args.iters, "phase_runtime": "phase_hetero_mytest", "execution": execution}, indent=2) + "\n")
>>>>>>> Stashed changes


def run_single_process(args: argparse.Namespace, scenario: Scenario) -> None:
    configure_runtime(args)
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization import phase_hetero_mytest
    import torch

    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    overrides = {"max_position_embeddings": scenario.input_seq + scenario.output_seq} if scenario.input_seq + scenario.output_seq > int(getattr(config, "max_position_embeddings", 0) or 0) else None
    prompts = make_prompts(scenario, 32000, args.seed, TokensPrompt, torch)
    output_seq = 1 if args.single_phase == "ttft" else scenario.output_seq
    runner = timed_reuse_phase_runs if args.reuse_llm else timed_phase_runs
    summary, rows = runner(args, scenario, output_seq, prompts, LLM, SamplingParams, phase_hetero_mytest, torch, overrides, args.single_phase)
    args.single_output.parent.mkdir(parents=True, exist_ok=True)
    args.single_output.write_text(json.dumps({"elapsed_ms": summary["mean_ms"]}) + "\n")
    if args.single_samples_dir:
        args.single_samples_dir.mkdir(parents=True, exist_ok=True)
        measured = [row for row in rows if not row["warmup"]]
        for index, row in enumerate(measured):
            (args.single_samples_dir / f"measured_{index}.json").write_text(
                json.dumps({"elapsed_ms": row["elapsed_ms"]}) + "\n")


def run_isolated_phase(args: argparse.Namespace, scenario: Scenario, output_seq: int, phase: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values, rows = [], []
    for index in range(args.warmup_iters + args.iters):
        result = args.output_dir / f".{phase}_{index}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint), "--scenario", scenario.name, "--output-dir", str(args.output_dir), "--vllm-root", str(args.vllm_root), "--cutlass-wrapper-path", str(args.cutlass_wrapper_path), "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--seed", str(args.seed), "--warmup-iters", "0", "--iters", "1", "--single-phase", phase, "--single-output", str(result)]
        command.extend(["--batch", str(scenario.batch), "--input-seq",
                        str(scenario.input_seq), "--output-seq",
                        str(scenario.output_seq)])
        if args.reuse_llm:
            command.append("--reuse-llm")
        if args.kv_cache_memory_bytes is not None:
            command.extend(["--kv-cache-memory-bytes",
                            str(args.kv_cache_memory_bytes)])
        if args.kv_cache_dtype is not None:
            command.extend(["--kv-cache-dtype", args.kv_cache_dtype])
        subprocess.run(command, check=True)
        elapsed = float(json.loads(result.read_text())["elapsed_ms"])
        result.unlink()
        warmup = index < args.warmup_iters
        rows.append({"method": "ours_max_speed", "scenario": scenario.name, "phase": phase, "iteration": index, "warmup": warmup, "batch": scenario.batch, "input_seq": scenario.input_seq, "output_seq": output_seq, "elapsed_ms": elapsed})
        if not warmup:
            values.append(elapsed)
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values)}, rows


def run_reuse_benchmark(args: argparse.Namespace, scenario: Scenario) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    """Match the baseline lifecycle: one LLM for TTFT and main measurements."""
    configure_runtime(args)
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization import phase_hetero_mytest
    import torch

    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    overrides = {"max_position_embeddings": scenario.input_seq + scenario.output_seq} if scenario.input_seq + scenario.output_seq > int(getattr(config, "max_position_embeddings", 0) or 0) else None
    prompts = make_prompts(scenario, 32000, args.seed, TokensPrompt, torch)
    llm = LLM(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=scenario.input_seq + scenario.output_seq,
              max_num_seqs=scenario.batch,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=True, enable_prefix_caching=False,
              enable_chunked_prefill=False,
              max_num_batched_tokens=scenario.batch * scenario.input_seq,
              hf_overrides=overrides)
    phase_hetero_mytest.enable_phase_hetero()
    try:
        ttft, ttft_rows = timed_reuse_phase_with_llm(args, scenario, 1, "ttft", prompts, llm, SamplingParams, phase_hetero_mytest, torch)
        phase_hetero_mytest.prepare_next_prefill()
        phase_hetero_mytest.wait_for_prefill_ready()
        main_stats, main_rows = timed_reuse_phase_with_llm(args, scenario, scenario.output_seq, "main", prompts, llm, SamplingParams, phase_hetero_mytest, torch)
        return ttft, ttft_rows, main_stats, main_rows
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def timed_reuse_phase_with_llm(args: Any, scenario: Scenario, output_seq: int, phase: str, prompts: list[Any], llm: Any, sampling_type: Any, phase_runtime: Any, torch: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values, rows = [], []
    sampling = sampling_type(max_tokens=output_seq, min_tokens=output_seq,
                             temperature=0.0, ignore_eos=True, detokenize=False)
    total = args.warmup_iters + args.iters
    for index in range(total):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        generated = sum(len(output.outputs[0].token_ids) for output in outputs)
        del outputs
        warmup = index < args.warmup_iters
        rows.append({"method": "ours_max_speed", "scenario": scenario.name,
                     "phase": phase, "iteration": index, "warmup": warmup,
                     "batch": scenario.batch, "input_seq": scenario.input_seq,
                     "output_seq": output_seq, "elapsed_ms": elapsed,
                     "generated_tokens": generated})
        if not warmup:
            values.append(elapsed)
        if index + 1 < total:
            phase_runtime.prepare_next_prefill()
            phase_runtime.wait_for_prefill_ready()
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values)}, rows


def configure_runtime(args: argparse.Namespace) -> None:
    sys.path[:0] = [str(args.vllm_root / "vllm"), str(args.vllm_root), str(args.cutlass_wrapper_path)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["PHASE_SWITCH_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)


def make_prompts(scenario: Scenario, vocab: int, seed: int, tokens_prompt: Any, torch: Any) -> list[Any]:
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed + scenario.batch * 1000003 + scenario.input_seq)
    ids = torch.randint(100, min(vocab, 30000), (scenario.batch, scenario.input_seq), generator=generator, dtype=torch.int64)
    return [tokens_prompt(prompt_token_ids=row.tolist()) for row in ids]


def timed_phase_runs(args: Any, scenario: Scenario, output_seq: int, prompts: list[Any], llm_type: Any, sampling_type: Any, phase_runtime: Any, torch: Any, overrides: Any, phase: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values, rows = [], []
    sampling = sampling_type(max_tokens=output_seq, min_tokens=output_seq, temperature=0.0, ignore_eos=True, detokenize=False)
    for index in range(args.warmup_iters + args.iters):
        llm_kwargs = dict(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1, max_model_len=scenario.input_seq + scenario.output_seq, max_num_seqs=scenario.batch, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=scenario.batch * scenario.input_seq, hf_overrides=overrides)
        if args.kv_cache_memory_bytes is not None:
            llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
        if args.kv_cache_dtype is not None:
            llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
        llm = llm_type(**llm_kwargs)
        phase_runtime.enable_phase_hetero()
        torch.cuda.synchronize(); started = time.perf_counter(); outputs = llm.generate(prompts, sampling, use_tqdm=False); torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        del outputs
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        is_warmup = index < args.warmup_iters
        rows.append({"method": "ours_max_speed", "scenario": scenario.name, "phase": phase, "iteration": index, "warmup": is_warmup, "batch": scenario.batch, "input_seq": scenario.input_seq, "output_seq": output_seq, "elapsed_ms": elapsed})
        if not is_warmup: values.append(elapsed)
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values)}, rows


def timed_reuse_phase_runs(args: Any, scenario: Scenario, output_seq: int,
                          prompts: list[Any], llm_type: Any, sampling_type: Any,
                          phase_runtime: Any, torch: Any, overrides: Any,
                          phase: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Measure consecutive requests after restoring prefill weights in-place."""
    values, rows = [], []
    sampling = sampling_type(max_tokens=output_seq, min_tokens=output_seq,
                             temperature=0.0, ignore_eos=True, detokenize=False)
    llm_kwargs = dict(model=str(args.checkpoint), dtype="bfloat16",
                   tensor_parallel_size=1,
                   max_model_len=scenario.input_seq + scenario.output_seq,
                   max_num_seqs=scenario.batch,
                   gpu_memory_utilization=args.gpu_memory_utilization,
                   enforce_eager=True, enable_prefix_caching=False,
                   enable_chunked_prefill=False,
                   max_num_batched_tokens=scenario.batch * scenario.input_seq,
                   hf_overrides=overrides)
    if args.kv_cache_memory_bytes is not None:
        llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.kv_cache_dtype is not None:
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    llm = llm_type(**llm_kwargs)
    phase_runtime.enable_phase_hetero()
    try:
        total = args.warmup_iters + args.iters
        for index in range(total):
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            del outputs
            warmup = index < args.warmup_iters
            rows.append({"method": "ours_max_speed", "scenario": scenario.name,
                         "phase": phase, "iteration": index, "warmup": warmup,
                         "batch": scenario.batch, "input_seq": scenario.input_seq,
                         "output_seq": output_seq, "elapsed_ms": elapsed})
            if not warmup:
                values.append(elapsed)
            if index + 1 < total:
                phase_runtime.prepare_next_prefill()
                phase_runtime.wait_for_prefill_ready()
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values)}, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
