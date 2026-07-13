#!/usr/bin/env python3
"""Run baseline-compatible PMPD quality evaluation for a phase-hetero checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS_ROOT = REPO_ROOT / "fake/kernels/cutlass/cutlass_wrapper"
DATA_ROOT = Path("/home/agent/wja/data/datasets/flaxquant")
BERTSCORE_MODEL = Path("/home/agent/wja/data/models/bert_score/roberta-large")
IWSLT_TOKENIZER = Path("/home/agent/wja/data/models/lmsys/vicuna-7b-v1.5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cnn_dm_1000", "dsum", "IWSLT"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--vllm-root", type=Path, default=VLLM_ROOT)
    parser.add_argument("--cutlass-wrapper-path", type=Path, default=CUTLASS_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=3840)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--bertscore-model", type=Path, default=BERTSCORE_MODEL)
    parser.add_argument("--iwslt-filter-tokenizer", default=str(IWSLT_TOKENIZER))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    sys.path.insert(0, str(REPO_ROOT / "references/pmpd_eval_kit"))
    import pmpd_eval
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.quantization import phase_hetero_mytest

    questions = pmpd_eval.build_questions(to_pmpd_args(args))
    target = args.output_dir / args.dataset
    target.mkdir(parents=True, exist_ok=True)
    write_config(args, target, len(questions))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=True)
    path = target / "ours_max_speed-vllm.jsonl"
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    with path.open("w", encoding="utf-8") as writer:
        for index, question in enumerate(questions, start=1):
            prompt = truncate_prompt(tokenizer, question["prompt"], args.max_input_tokens)
            llm = LLM(model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1, enforce_eager=True, skip_mm_profiling=True, gpu_memory_utilization=args.gpu_memory_utilization, enable_chunked_prefill=False, max_num_batched_tokens=args.max_input_tokens, max_model_len=args.max_input_tokens + args.max_new_tokens)
            phase_hetero_mytest.enable_phase_hetero()
            started = time.perf_counter(); output = llm.generate([prompt], sampling, use_tqdm=False)[0]; elapsed = time.perf_counter() - started
            choice = output.outputs[0]
            writer.write(json.dumps({"question_id": question["question_id"], "answer_id": f"vllm-ours_max_speed-{question['question_id']}", "model_id": "ours_max_speed", "choices": [{"index": 0, "turns": [choice.text], "idxs": [max(len(choice.token_ids) - 1, 0)], "new_tokens": [len(choice.token_ids)], "wall_time": [elapsed], "precision_log": [{"16": len(choice.token_ids)}]}], "reference": question["reference"], "tstamp": time.time()}, ensure_ascii=False) + "\n")
            del output
            llm.llm_engine.engine_core.shutdown()
            del llm; gc.collect()
            try:
                import torch
                torch.cuda.empty_cache(); torch.cuda.synchronize()
            except Exception:
                pass
            if index == len(questions) or index % args.log_every == 0:
                print(f"[progress] {args.dataset}: {index}/{len(questions)}", flush=True)
    pmpd_eval.compute_metrics(to_pmpd_args(args), path)


def configure_runtime(args: argparse.Namespace) -> None:
    sys.path[:0] = [str(args.vllm_root / "vllm"), str(args.vllm_root), str(args.cutlass_wrapper_path)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)
    os.environ["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)


def truncate_prompt(tokenizer: Any, prompt: str, max_input_tokens: int) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    return prompt if len(ids) <= max_input_tokens else tokenizer.decode(ids[-max_input_tokens:], skip_special_tokens=True)


def to_pmpd_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(dataset=args.dataset, split="test", data_root=args.data_root, model_path=args.checkpoint, model_id="ours_max_speed", output_dir=args.output_dir, max_new_tokens=args.max_new_tokens, max_input_tokens=args.max_input_tokens, question_begin=None, question_end=None, log_every=args.log_every, bertscore_model=args.bertscore_model, bertscore_num_layers=17, iwslt_filter_tokenizer=args.iwslt_filter_tokenizer, metrics_only=None)


def write_config(args: argparse.Namespace, target: Path, count: int) -> None:
    target.joinpath("run_config.json").write_text(json.dumps({"backend": "vllm", "phase_hetero": True, "method": "ours_max_speed", "dataset": args.dataset, "num_questions": count, "model_path": str(args.checkpoint), "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens}, indent=2) + "\n")


if __name__ == "__main__":
    main()
