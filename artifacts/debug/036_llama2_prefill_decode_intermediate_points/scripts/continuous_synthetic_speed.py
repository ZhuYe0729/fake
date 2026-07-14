#!/usr/bin/env python3
"""Fixed-length continuous phase-hetero speed diagnostic (one vLLM load)."""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    vllm_root = Path("/home/agent/wja/project/my/cospaq/test/vllm")
    phase_root = vllm_root / "artifacts/dev/012_phase_hetero_linear"
    cutlass = Path("/root/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper")
    sys.path[:0] = [str(vllm_root / "vllm"), str(vllm_root), str(phase_root), str(cutlass)]
    from vllm import LLM, SamplingParams
    import torch

    phase = importlib.import_module("phase_hetero_mytest")
    llm = LLM(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=2128, max_num_seqs=16, max_num_batched_tokens=32768,
              gpu_memory_utilization=0.85, enforce_eager=True, enable_prefix_caching=False,
              enable_chunked_prefill=False)
    phase.enable_phase_hetero()
    prompts = [{"prompt_token_ids": [1] * 2048} for _ in range(16)]
    sampling = SamplingParams(max_tokens=80, min_tokens=80, temperature=0.0, ignore_eos=True, detokenize=False)
    rows = []
    for index in range(args.warmup + args.iters):
        if index:
            phase.wait_for_prefill_ready()
        torch.cuda.synchronize(); started = time.perf_counter()
        llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize(); elapsed = (time.perf_counter() - started) * 1000.0
        if index + 1 < args.warmup + args.iters:
            phase.prepare_next_prefill()
        rows.append({"iteration": index, "warmup": index < args.warmup, "elapsed_ms": elapsed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
