#!/usr/bin/env python3
"""Lightweight quality checks for phase-switch experiments."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


EXAMPLES = [
    ("The capital of France is", " Paris."),
    ("The largest planet in the solar system is", " Jupiter."),
    ("Machine learning is a field of study that gives computers",
     " the ability to learn from data."),
    ("In a neural network, the parameters are updated using",
     " gradient descent."),
    ("The chemical symbol for water is", " H2O."),
    ("The opposite of hot is", " cold."),
    ("A compiler translates source code into", " machine code."),
    ("The Pacific Ocean is larger than", " the Atlantic Ocean."),
]

GEN_PROMPTS = [
    "The capital of France is",
    "In machine learning, quantization is",
    "A short explanation of gravity:",
    "A transformer language model predicts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--cutlass-wrapper-path", required=True)
    parser.add_argument("--phase-switch", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def token_logprob(entry: Any, token_id: int) -> float:
    if entry is None:
        raise ValueError("missing prompt logprob entry")
    if token_id not in entry:
        obj = next(iter(entry.values()))
    else:
        obj = entry[token_id]
    value = getattr(obj, "logprob", obj)
    return float(value)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo_root) / "vllm"))
    sys.path.insert(0, str(Path(args.artifact_dir)))
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

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    llm = LLM(model=args.model,
              dtype=args.dtype,
              trust_remote_code=True,
              enforce_eager=True,
              tensor_parallel_size=1,
              skip_mm_profiling=True)

    if phase_switch_mytest is not None:
        phase_switch_mytest.enable_phase_switch()

    full_texts = [prefix + continuation for prefix, continuation in EXAMPLES]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(full_texts, params)

    total_nll = 0.0
    total_tokens = 0
    records = []
    for (prefix, continuation), output in zip(EXAMPLES, outputs):
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
        full_ids = tokenizer.encode(prefix + continuation,
                                    add_special_tokens=True)
        if output.prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt_logprobs")
        token_nlls = []
        for idx in range(len(prefix_ids), len(full_ids)):
            lp = token_logprob(output.prompt_logprobs[idx], full_ids[idx])
            token_nlls.append(-lp)
        nll = sum(token_nlls)
        total_nll += nll
        total_tokens += len(token_nlls)
        records.append({
            "prefix": prefix,
            "continuation": continuation,
            "tokens": len(token_nlls),
            "nll": nll,
            "avg_nll": nll / max(1, len(token_nlls)),
        })

    gen_params = SamplingParams(temperature=0.0, max_tokens=64)
    gen_outputs = llm.generate(GEN_PROMPTS, gen_params)
    generations = []
    for item in gen_outputs:
        generations.append({
            "prompt": item.prompt,
            "output": item.outputs[0].text,
            "token_ids": list(item.outputs[0].token_ids),
        })

    avg_nll = total_nll / total_tokens
    result = {
        "label": args.label,
        "model": args.model,
        "num_examples": len(EXAMPLES),
        "num_scored_tokens": total_tokens,
        "total_nll": total_nll,
        "avg_nll": avg_nll,
        "perplexity": math.exp(avg_nll),
        "examples": records,
        "generations": generations,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) +
                           "\n")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
