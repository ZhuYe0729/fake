#!/usr/bin/env python3
"""Fresh-process phase-vLLM prefill benchmark with an explicit scheduler cap."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS_ROOT = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-seq", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    sys.path[:0] = [str(VLLM_ROOT / "vllm"), str(VLLM_ROOT), str(CUTLASS_ROOT)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS_ROOT)

    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization import phase_hetero_mytest

    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    total_seq = args.input_seq + 1
    overrides = ({"max_position_embeddings": total_seq}
                 if int(getattr(config, "max_position_embeddings", 0) or 0) < total_seq
                 else None)
    llm = LLM(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=total_seq, max_num_seqs=args.batch,
              max_num_batched_tokens=args.max_num_batched_tokens,
              gpu_memory_utilization=0.90, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              hf_overrides=overrides)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.batch * 1_000_003 + args.input_seq)
    tokens = torch.randint(100, 30_000, (args.batch, args.input_seq),
                           generator=generator, dtype=torch.int64)
    prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in tokens]
    sampling = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0,
                              ignore_eos=True, detokenize=False)
    phase_hetero_mytest.enable_phase_hetero()
    def run_once(prepare_prefill: bool) -> tuple[float, int]:
        if prepare_prefill:
            phase_hetero_mytest.prepare_next_prefill()
            phase_hetero_mytest.wait_for_prefill_ready()
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        return ((time.perf_counter() - started) * 1000.0,
                sum(len(item.outputs[0].token_ids) for item in outputs))

    # A newly enabled runtime starts in prefill; the API only permits an
    # explicit prepare after the preceding request has completed its decode.
    warmup_ms = [run_once(index > 0)[0] for index in range(args.warmup)]
    timed = [run_once(True) for _ in range(args.repeats)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "batch": args.batch,
        "input_seq": args.input_seq,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.batch,
        "warmup_requests": args.warmup,
        "warmup_ms": warmup_ms,
        "timed_ms": [item[0] for item in timed],
        "median_ms": statistics.median(item[0] for item in timed),
        "generated_tokens": [item[1] for item in timed],
        "protocol": "same_engine_warmup_then_timed_explicit_prefill_phase",
    }, indent=2) + "\n")
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
