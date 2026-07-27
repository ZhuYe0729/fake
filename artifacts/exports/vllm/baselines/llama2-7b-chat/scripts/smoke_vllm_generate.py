#!/usr/bin/env python3
"""Run one deterministic vLLM generation as a local checkpoint smoke test."""

from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="[INST] Explain in one sentence what quantized inference is. [/INST]",
    )
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=512,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
