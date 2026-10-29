#!/usr/bin/env python3
"""Run offline vLLM generate with artifact-local phase-switch quantization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--cutlass-wrapper-path", required=True)
    return parser.parse_args()


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

    import phase_switch_mytest
    from vllm import LLM, SamplingParams

    prompts = [
        "The capital of France is",
        "A GPU kernel is",
    ]
    llm = LLM(model=args.model,
              dtype="bfloat16",
              trust_remote_code=True,
              enforce_eager=True,
              skip_mm_profiling=True,
              enable_chunked_prefill=False)

    phase_switch_mytest.enable_phase_switch()
    outputs = llm.generate(prompts,
                           SamplingParams(max_tokens=16, temperature=0.0))

    rows = []
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text
        if not text:
            raise AssertionError(f"empty output for prompt {prompt!r}")
        rows.append({"prompt": prompt, "output": text})

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    phase_switch_mytest.dump_trace(args.trace_json)


if __name__ == "__main__":
    main()
