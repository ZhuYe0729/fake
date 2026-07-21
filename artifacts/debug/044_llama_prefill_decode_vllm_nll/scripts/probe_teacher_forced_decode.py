#!/usr/bin/env python3
"""Probe teacher-forced decode NLL through a real vLLM generation request."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[4]
VLLM = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--continuation", default=" Paris, and it is known for art and history.")
    parser.add_argument("--disable-processor", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(VLLM / "vllm"))
    sys.path.insert(0, str(VLLM))
    sys.path.insert(0, str(CUTLASS))
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)

    from teacher_force_processor import TeacherForceBatchProcessor
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    target_ids = tokenizer.encode(args.continuation, add_special_tokens=False)
    capture_path = args.output.with_suffix(".capture.jsonl")
    capture_path.unlink(missing_ok=True)
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              enable_prefix_caching=False, skip_mm_profiling=True,
              max_model_len=2048,
              logits_processors=[] if args.disable_processor else [TeacherForceBatchProcessor])
    params = SamplingParams(
        temperature=0.0,
        max_tokens=len(target_ids),
        extra_args=None if args.disable_processor else {
            "teacher_force_target_ids": target_ids,
            "teacher_force_capture_path": str(capture_path)})
    output = llm.generate([args.prompt], params, use_tqdm=False)[0]
    generated = list(output.outputs[0].token_ids)
    if args.disable_processor:
        result = {"model": args.model, "processor_disabled": True,
                  "generated_ids": generated}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if generated != target_ids:
        raise AssertionError("teacher forcing did not reproduce target ids")
    records = [json.loads(line) for line in capture_path.read_text().splitlines()]
    logprobs = [record["logprob"] for record in records]
    if len(logprobs) != len(target_ids):
        raise AssertionError(f"expected {len(target_ids)} captured logprobs, got {len(logprobs)}")
    result = {"model": args.model, "prompt": args.prompt,
              "target_ids": target_ids, "generated_ids": generated,
              "token_count": len(target_ids), "avg_nll": -sum(logprobs) / len(logprobs),
              "perplexity": math.exp(-sum(logprobs) / len(logprobs)),
              "logprobs": logprobs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
