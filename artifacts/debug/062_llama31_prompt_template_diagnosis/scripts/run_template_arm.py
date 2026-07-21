#!/usr/bin/env python3
"""Run one dense-BF16 Llama3 PMPD prompt-template diagnosis arm."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/062_llama31_prompt_template_diagnosis"
MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
DATA_ROOT = Path("/home/agent/wja/data/datasets/flaxquant")
IWSLT_FILTER_TOKENIZER = "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"
TASKS = ("cnn_dm_1000", "dsum", "IWSLT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", choices=("legacy", "native"), required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def user_message(legacy_prompt: str) -> str:
    prefix = "Play the role of assistant and answer the question from human. Human: "
    suffix = "\n\nAssistant: "
    if not legacy_prompt.startswith(prefix) or not legacy_prompt.endswith(suffix):
        raise ValueError("unexpected PMPD legacy prompt format")
    return legacy_prompt[len(prefix):-len(suffix)]


def trim(tokens: list[int], limit: int) -> list[int]:
    return tokens if len(tokens) <= limit else tokens[-limit:]


def marker_flags(text: str) -> list[str]:
    markers = ("Human:", "Assistant:", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
    return [marker for marker in markers if marker in text]


def main() -> None:
    args = parse_args()
    sys.path[:0] = [str(VLLM_ROOT / "vllm"), str(VLLM_ROOT), str(ROOT / "references")]
    os.environ.update({"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                       "VLLM_USE_FLASHINFER_SAMPLER": "0", "TOKENIZERS_PARALLELISM": "false",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    from pmpd_eval_kit import pmpd_eval
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_ids = sorted({item for item in (tokenizer.eos_token_id, eot_id) if item is not None and item >= 0})
    output_root = EXP / "outputs" / args.template
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"template": args.template, "model": str(MODEL), "samples_per_task": args.samples,
                "max_new_tokens": 256, "max_input_tokens": 3840, "batch_size": args.batch_size,
                "stop_token_ids": stop_ids, "eos_token_id": tokenizer.eos_token_id, "eot_token_id": eot_id,
                "legacy_template": "PMPD role-play + Human/Assistant", "native_template": "apply_chat_template(user, add_generation_prompt=True)"}
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    llm = LLM(model=str(MODEL), dtype="bfloat16", trust_remote_code=True, enforce_eager=True,
              skip_mm_profiling=True, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=4096, max_num_batched_tokens=15360, enable_prefix_caching=False,
              enable_chunked_prefill=False)
    sampling = SamplingParams(max_tokens=256, temperature=0.0, stop_token_ids=stop_ids)
    for task in TASKS:
        question_args = SimpleNamespace(dataset=task, split="test", data_root=DATA_ROOT,
                                        iwslt_filter_tokenizer=IWSLT_FILTER_TOKENIZER, question_begin=0,
                                        question_end=args.samples)
        questions = pmpd_eval.build_questions(question_args)
        if len(questions) != args.samples:
            raise RuntimeError(f"{task}: expected {args.samples}, got {len(questions)}")
        records, prompts = [], []
        for question in questions:
            legacy = question["prompt"]
            if args.template == "legacy":
                token_ids = tokenizer.encode(legacy, add_special_tokens=True)
            else:
                token_ids = tokenizer.apply_chat_template([{"role": "user", "content": user_message(legacy)}],
                                                           tokenize=True, add_generation_prompt=True)
            token_ids = trim(list(token_ids), 3840)
            prompts.append({"prompt_token_ids": token_ids})
            records.append({"question_id": question["question_id"], "reference": question["reference"],
                            "input_tokens": len(token_ids), "prompt_preview": tokenizer.decode(token_ids[-160:], skip_special_tokens=False)})
        target = output_root / task / "answers.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with target.open("w", encoding="utf-8") as handle:
            for begin in range(0, len(prompts), args.batch_size):
                outputs = llm.generate(prompts[begin:begin + args.batch_size], sampling, use_tqdm=False)
                for record, output in zip(records[begin:begin + args.batch_size], outputs):
                    answer = output.outputs[0]
                    raw = tokenizer.decode(answer.token_ids, skip_special_tokens=False)
                    record.update({"text": answer.text, "raw_decoded": raw, "new_tokens": len(answer.token_ids),
                                   "finish_reason": answer.finish_reason, "stop_reason": answer.stop_reason,
                                   "role_marker_continuations": marker_flags(raw)})
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{args.template}] {task}: {min(begin + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
        (output_root / task / "timing.json").write_text(json.dumps({"wall_seconds": time.perf_counter() - started}) + "\n")


if __name__ == "__main__":
    main()
