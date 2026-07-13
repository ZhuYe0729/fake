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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, default=VLLM_ROOT)
    parser.add_argument("--cutlass-wrapper-path", type=Path, default=CUTLASS_ROOT)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single-phase", choices=("ttft", "main"))
    parser.add_argument("--single-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = Scenario(args.scenario, *SCENARIOS[args.scenario])
    if args.single_phase:
        run_single_process(args, scenario)
        return
    output_dir = args.output_dir; output_dir.mkdir(parents=True, exist_ok=True)
    ttft, ttft_rows = run_isolated_phase(args, scenario, 1, "ttft")
    main_stats, main_rows = run_isolated_phase(args, scenario, scenario.output_seq, "main")
    tpot = 0.0 if scenario.output_seq <= 1 else (main_stats["median_ms"] - ttft["median_ms"]) / (scenario.output_seq - 1)
    summary = {"method": "ours_max_speed", "scenario": scenario.name, "checkpoint": str(args.checkpoint), "status": "OK",
               "batch": scenario.batch, "input_seq": scenario.input_seq, "output_seq": scenario.output_seq,
               "e2e_mean_ms": main_stats["mean_ms"], "e2e_median_ms": main_stats["median_ms"], "ttft_median_ms": ttft["median_ms"],
               "tpot_ms": tpot, "total_tokens_per_s": scenario.batch * (scenario.input_seq + scenario.output_seq) * 1000.0 / main_stats["mean_ms"]}
    write_csv(output_dir / "iterations.csv", ttft_rows + main_rows)
    write_csv(output_dir / "summary.csv", [summary])
    (output_dir / "metadata.json").write_text(json.dumps({"scenario": scenario.__dict__, "warmup_iters": args.warmup_iters, "iters": args.iters, "phase_runtime": "phase_hetero_mytest", "execution": "one_vllm_process_per_sample"}, indent=2) + "\n")


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
    summary, _rows = timed_phase_runs(args, scenario, output_seq, prompts, LLM, SamplingParams, phase_hetero_mytest, torch, overrides, args.single_phase)
    args.single_output.write_text(json.dumps({"elapsed_ms": summary["mean_ms"]}) + "\n")


def run_isolated_phase(args: argparse.Namespace, scenario: Scenario, output_seq: int, phase: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values, rows = [], []
    for index in range(args.warmup_iters + args.iters):
        result = args.output_dir / f".{phase}_{index}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint), "--scenario", scenario.name, "--output-dir", str(args.output_dir), "--vllm-root", str(args.vllm_root), "--cutlass-wrapper-path", str(args.cutlass_wrapper_path), "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--seed", str(args.seed), "--warmup-iters", "0", "--iters", "1", "--single-phase", phase, "--single-output", str(result)]
        subprocess.run(command, check=True)
        elapsed = float(json.loads(result.read_text())["elapsed_ms"])
        result.unlink()
        warmup = index < args.warmup_iters
        rows.append({"method": "ours_max_speed", "scenario": scenario.name, "phase": phase, "iteration": index, "warmup": warmup, "batch": scenario.batch, "input_seq": scenario.input_seq, "output_seq": output_seq, "elapsed_ms": elapsed})
        if not warmup:
            values.append(elapsed)
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
        llm = llm_type(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1, max_model_len=scenario.input_seq + scenario.output_seq, max_num_seqs=scenario.batch, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=scenario.batch * scenario.input_seq, hf_overrides=overrides)
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
