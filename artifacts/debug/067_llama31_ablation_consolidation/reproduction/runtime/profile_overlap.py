#!/usr/bin/env python3
"""Measure copy work and compute-stream stalls for one overlap execution."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
CHECKPOINT = EXP / "checkpoint/point_011"
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path[:0] = [str(VLLM_ROOT / "vllm"), str(VLLM_ROOT), str(CUTLASS)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["PHASE_SWITCH_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)

    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization import phase_hetero_mytest

    batch, input_seq, output_seq = 8, 2048, 64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(batch * 1000003 + input_seq)
    token_ids = torch.randint(100, 30000, (batch, input_seq),
                              generator=generator, dtype=torch.int64)
    prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in token_ids]
    sampling = SamplingParams(max_tokens=output_seq, min_tokens=output_seq,
                              temperature=0.0, ignore_eos=True,
                              detokenize=False)
    llm = LLM(model=str(CHECKPOINT), dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=input_seq + output_seq, max_num_seqs=batch,
              gpu_memory_utilization=0.80, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              max_num_batched_tokens=batch * input_seq)

    copy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    wait_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    original_materialize = phase_hetero_mytest._materialize_decode_from_cpu
    original_wait = phase_hetero_mytest._wait_decode_ready

    def profiled_materialize(layer: torch.nn.Module,
                             non_blocking: bool) -> None:
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        original_materialize(layer, non_blocking)
        end.record(stream)
        copy_events.append((start, end))

    def profiled_wait(layer: torch.nn.Module) -> None:
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        original_wait(layer)
        end.record(stream)
        wait_events.append((start, end))

    phase_hetero_mytest._materialize_decode_from_cpu = profiled_materialize
    phase_hetero_mytest._wait_decode_ready = profiled_wait
    phase_hetero_mytest.enable_phase_hetero()
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - started) * 1000.0

    copy_ms = sum(start.elapsed_time(end) for start, end in copy_events)
    exposed_ms = sum(start.elapsed_time(end) for start, end in wait_events)
    hidden_ms = max(0.0, copy_ms - exposed_ms)
    hidden_ratio = hidden_ms / copy_ms if copy_ms else 0.0
    result = {
        "e2e_latency_ms": e2e_ms,
        "total_transfer_work_ms": copy_ms,
        "exposed_copy_stall_ms": exposed_ms,
        "hidden_transfer_work_ms": hidden_ms,
        "hidden_ratio": hidden_ratio,
        "materialized_linear_modules": len(copy_events),
        "first_decode_waits": len(wait_events),
        "model": "Llama3.1-8B-Instruct",
        "policy": "point_011",
        "scenario": {"batch": batch, "input_seq": input_seq,
                     "output_seq": output_seq},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    del outputs
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()

