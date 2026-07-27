#!/usr/bin/env python3
"""PMPD-style evaluation using vLLM generation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn_dm_1000",
                        choices=["cnn_dm", "cnn_dm_1000", "dsum", "IWSLT"])
    parser.add_argument("--split", default="test")
    from common import BERTSCORE_MODEL, IWSLT_FILTER_TOKENIZER, PMPD, PMPD_DATA_ROOT
    parser.add_argument("--data-root", type=Path, default=PMPD_DATA_ROOT)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=PMPD["max_new_tokens"])
    parser.add_argument("--max-input-tokens", type=int, default=PMPD["max_input_tokens"])
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--batch-size", type=int, default=PMPD["batch_size"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--bertscore-model", type=Path, default=BERTSCORE_MODEL)
    parser.add_argument("--bertscore-num-layers", type=int, default=17)
    parser.add_argument("--iwslt-filter-tokenizer", default=str(IWSLT_FILTER_TOKENIZER))
    parser.add_argument("--repo-root")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--cutlass-wrapper-path")
    parser.add_argument("--phase-switch", action="store_true")
    parser.add_argument("--phase-hetero", action="store_true")
    parser.add_argument("--phase-artifact-dir")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--append", action="store_true",
                        help="Append answers instead of replacing the output file.")
    return parser.parse_args()


def answer_path(args: argparse.Namespace) -> Path:
    split_suffix = "" if args.split == "test" else f"_{args.split}"
    return (args.output_dir / args.dataset /
            f"{args.model_id}-fp16{split_suffix}.jsonl")


def truncate_prompt(tokenizer: Any, prompt: str, max_input_tokens: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(token_ids) <= max_input_tokens:
        return prompt
    return tokenizer.decode(token_ids[-max_input_tokens:],
                            skip_special_tokens=True)


def write_run_config(args: argparse.Namespace, output_dir: Path,
                     num_questions: int) -> None:
    config = {
        "dataset": args.dataset,
        "split": args.split,
        "num_questions": num_questions,
        "model_path": str(args.model_path),
        "model_id": args.model_id,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "question_begin": args.question_begin,
        "question_end": args.question_end,
        "batch_size": args.batch_size,
        "backend": "vllm",
        "phase_switch": args.phase_switch,
        "phase_hetero": args.phase_hetero,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main() -> None:
    args = parse_args()
    from common import CUTLASS, VLLM_ROOT
    from vllm_compat import assert_chunked_prefill_disabled, force_v1_chunked_prefill_disabled
    repo_root = Path(args.repo_root) if args.repo_root else VLLM_ROOT
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else Path(__file__).parent
    args.cutlass_wrapper_path = args.cutlass_wrapper_path or str(CUTLASS)
    del args.phase_artifact_dir
    if args.phase_switch and args.phase_hetero:
        raise ValueError("--phase-switch and --phase-hetero are exclusive")
    sys.path.insert(0, str(repo_root / "vllm"))
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(artifact_dir))
    sys.path.insert(0, args.cutlass_wrapper_path)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("PHASE_SWITCH_TRACE", "0")
    os.environ.setdefault("PHASE_SWITCH_RELEASE_PREFILL", "0")
    os.environ.setdefault("PHASE_HETERO_TRACE", "0")
    os.environ["PHASE_SWITCH_MYTEST_CUTLASS_WRAPPER_PATH"] = (
        args.cutlass_wrapper_path)
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = args.cutlass_wrapper_path
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = (
        args.cutlass_wrapper_path)

    import pmpd_eval
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    phase_switch_mytest = None
    phase_hetero_mytest = None
    if args.phase_switch:
        import phase_switch_mytest as _phase_switch_mytest
        phase_switch_mytest = _phase_switch_mytest
    if args.phase_hetero:
        from vllm.model_executor.layers.quantization import (
            phase_hetero_mytest as _phase_hetero_mytest)
        phase_hetero_mytest = _phase_hetero_mytest

    questions = pmpd_eval.build_questions(args)
    out_path = answer_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(args, out_path.parent, len(questions))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True,
                                              use_fast=True)
    prompts = [
        truncate_prompt(tokenizer, question["prompt"], args.max_input_tokens)
        for question in questions
    ]

    llm_kwargs = {
        "model": args.model_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "enforce_eager": True,
        "skip_mm_profiling": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.phase_hetero:
        llm_kwargs["enable_chunked_prefill"] = False
        # Phase-hetero requires a complete prefill before decode; a prefix
        # cache hit can otherwise skip that transition for a later batch.
        llm_kwargs["enable_prefix_caching"] = False
        if args.max_num_batched_tokens is None:
            raise ValueError("--phase-hetero requires --max-num-batched-tokens")
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    sampling = SamplingParams(max_tokens=args.max_new_tokens,
                              temperature=0.0)

    force_v1_chunked_prefill_disabled()
    llm = LLM(**llm_kwargs)
    assert_chunked_prefill_disabled(llm)
    if phase_hetero_mytest is not None:
        phase_hetero_mytest.enable_phase_hetero()

    started = time.perf_counter()
    write_mode = "a" if args.append else "w"
    with out_path.open(write_mode, encoding="utf-8") as writer:
        for batch_start in range(0, len(questions), args.batch_size):
            batch_questions = questions[batch_start:batch_start +
                                        args.batch_size]
            batch_prompts = prompts[batch_start:batch_start + args.batch_size]
            if phase_hetero_mytest is not None and batch_start:
                phase_hetero_mytest.wait_for_prefill_ready()
            if phase_switch_mytest is not None:
                phase_switch_mytest.enable_phase_switch()
            batch_wall_start = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
            batch_wall = time.perf_counter() - batch_wall_start
            if (phase_hetero_mytest is not None and
                    batch_start + len(batch_questions) < len(questions)):
                phase_hetero_mytest.prepare_next_prefill()
            for question, output in zip(batch_questions, outputs):
                token_ids = list(output.outputs[0].token_ids)
                new_token = len(token_ids)
                record = {
                    "question_id": question["question_id"],
                    "answer_id": f"vllm-{question['question_id']}",
                    "model_id": args.model_id,
                    "choices": [{
                        "index": 0,
                        "turns": [output.outputs[0].text],
                        "idxs": [max(new_token - 1, 0)],
                        "new_tokens": [new_token],
                        "wall_time": [batch_wall / max(1, len(batch_questions))],
                        "precision_log": [{"16": new_token}],
                    }],
                    "reference": question["reference"],
                    "tstamp": time.time(),
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            done = batch_start + len(batch_questions)
            if done == len(questions) or done % args.log_every == 0:
                elapsed = time.perf_counter() - started
                print(f"[progress] {args.model_id}: {done}/{len(questions)} "
                      f"wall_seconds={elapsed:.2f}",
                      flush=True)

    del llm
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    if not args.skip_metrics:
        metrics_path = pmpd_eval.compute_metrics(args, out_path)
        model_metrics_path = metrics_path.with_name(
            f"{args.model_id}_metrics.json")
        model_metrics_path.write_text(metrics_path.read_text(encoding="utf-8"),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
