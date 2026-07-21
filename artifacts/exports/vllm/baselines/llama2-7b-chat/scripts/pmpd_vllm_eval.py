#!/usr/bin/env python3
"""Run PMPD-style generation metrics with vLLM checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BASELINE_ROOT.parents[4]
PMPD_ROOT = REPO_ROOT / "references/pmpd_eval_kit"
if str(PMPD_ROOT) not in sys.path:
    sys.path.insert(0, str(PMPD_ROOT))

import pmpd_eval  # noqa: E402


MODEL_PATH = Path(
    os.environ.get("LLAMA2_7B_CHAT_MODEL_PATH", "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
)
DEFAULT_DATA_ROOT = Path("/home/agent/wja/data/datasets/flaxquant")
DEFAULT_BERTSCORE_MODEL = Path("/home/agent/wja/data/models/bert_score/roberta-large")
DEFAULT_IWSLT_FILTER_TOKENIZER = Path("/home/agent/wja/data/models/lmsys/vicuna-7b-v1.5")
METHOD_PATHS = {
    "dense_bf16": MODEL_PATH,
    "dense_nvfp4": BASELINE_ROOT / "checkpoints/uniform_dense_nvfp4",
    "sparse_bf16": BASELINE_ROOT / "checkpoints/uniform_sparse_bf16",
    "sparse_nvfp4": BASELINE_ROOT / "checkpoints/uniform_sparse_nvfp4",
    "marlin_nvfp4": BASELINE_ROOT / "checkpoints/uniform_marlin_nvfp4",
}
DATASETS = ("cnn_dm_1000", "dsum", "IWSLT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_PATHS), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT / "results/quality")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=3840)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--bertscore-model", type=Path, default=DEFAULT_BERTSCORE_MODEL)
    parser.add_argument("--bertscore-num-layers", type=int, default=17)
    parser.add_argument("--iwslt-filter-tokenizer", default=str(DEFAULT_IWSLT_FILTER_TOKENIZER))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--metrics-only", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    os.environ.pop("_CUDA_COMPAT_STATUS", None)
    args = parse_args()
    if args.metrics_only:
        pmpd_args = to_pmpd_args(args, METHOD_PATHS[args.method])
        pmpd_eval.compute_metrics(pmpd_args, args.metrics_only)
        return

    model_path = METHOD_PATHS[args.method]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    answer_file = run_generation(args, model_path)
    pmpd_eval.compute_metrics(to_pmpd_args(args, model_path), answer_file)


def run_generation(args: argparse.Namespace, model_path: Path) -> Path:
    pmpd_args = to_pmpd_args(args, model_path)
    questions = pmpd_eval.build_questions(pmpd_args)
    out_path = answer_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(args, model_path, out_path.parent, len(questions))

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=args.max_input_tokens + args.max_new_tokens,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    run_started = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as writer:
        for index, question in enumerate(questions, start=1):
            prompt = pmpd_eval.truncate_prompt(
                tokenizer, question["prompt"], args.max_input_tokens
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            torch.cuda.synchronize()
            wall_time = time.perf_counter() - started
            choice = outputs[0].outputs[0]
            output_text = choice.text
            new_token = len(choice.token_ids)
            record = {
                "question_id": question["question_id"],
                "answer_id": f"vllm-{args.method}-{question['question_id']}",
                "model_id": args.method,
                "choices": [
                    {
                        "index": 0,
                        "turns": [output_text],
                        "idxs": [max(new_token - 1, 0)],
                        "new_tokens": [new_token],
                        "wall_time": [wall_time],
                        "precision_log": [{"16": new_token}],
                    }
                ],
                "reference": question["reference"],
                "tstamp": time.time(),
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index == len(questions) or index % args.log_every == 0:
                elapsed = time.perf_counter() - run_started
                print(
                    f"[progress] {args.method}/{args.dataset}: {index}/{len(questions)} "
                    f"wall_seconds={elapsed:.2f}",
                    flush=True,
                )

    cleanup_cuda()
    return out_path


def answer_path(args: argparse.Namespace) -> Path:
    split_suffix = "" if args.split == "test" else f"_{args.split}"
    return args.output_dir / args.method / args.dataset / f"{args.method}-vllm{split_suffix}.jsonl"


def to_pmpd_args(args: argparse.Namespace, model_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=args.dataset,
        split=args.split,
        data_root=args.data_root,
        model_path=model_path,
        model_id=args.method,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        question_begin=args.question_begin,
        question_end=args.question_end,
        log_every=args.log_every,
        bertscore_model=args.bertscore_model,
        bertscore_num_layers=args.bertscore_num_layers,
        iwslt_filter_tokenizer=str(args.iwslt_filter_tokenizer),
        metrics_only=args.metrics_only,
    )


def write_run_config(
    args: argparse.Namespace, model_path: Path, output_dir: Path, num_questions: int
) -> None:
    config = {
        "backend": "vllm",
        "method": args.method,
        "dataset": args.dataset,
        "split": args.split,
        "num_questions": num_questions,
        "model_path": str(model_path),
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "question_begin": args.question_begin,
        "question_end": args.question_end,
        "bertscore_model": str(args.bertscore_model),
        "bertscore_num_layers": args.bertscore_num_layers,
        "iwslt_filter_tokenizer": str(args.iwslt_filter_tokenizer),
        "cnn_dm_note": (
            "cnn_dm_1000 is the fixed 1000-example subset, not full CNN/DM."
            if args.dataset == "cnn_dm_1000"
            else ""
        ),
        "pmpd_style": {
            "batch_size": 1,
            "decoding": "vllm greedy temperature=0",
            "prompt_template": "FastChat Claude-style ADD_COLON_SINGLE plus PMPD role-play prefix",
            "iwslt_direction": "French to English",
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def cleanup_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
