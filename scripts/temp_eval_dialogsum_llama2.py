#!/usr/bin/env python3
"""Temporary DialogSum generative evaluation for Llama-2-7B."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from datasets import DownloadConfig, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL_PATH = "/home/agent/wja/data/models/LLM-Research/llama-2-7b"
PROMPT_TEMPLATE = "Summarize the following dialogue.\n\n{dialogue}\n\nSummary:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Llama-2-7B on DialogSum with generation + ROUGE.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", default="knkarthick/dialogsum", help="HF dataset name or local dataset path.")
    parser.add_argument("--split", default="test", help="Preferred split; falls back to test if unavailable.")
    parser.add_argument("--output-dir", default="artifacts/results/dialogsum_llama2_7b/temp_eval")
    parser.add_argument("--limit", type=int, default=16, help="Limit examples for a quick smoke test; use -1 for full split.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default=None, help="Optional transformers attention implementation, e.g. sdpa.")
    parser.add_argument("--local-files-only", action="store_true", help="Require cached dataset/model files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_model(args)
    dataset_split, split_name = load_dialogsum(args)
    if args.limit is not None and args.limit >= 0:
        dataset_split = dataset_split.select(range(min(args.limit, len(dataset_split))))

    results_path = output_dir / "results.jsonl"
    predictions: list[str] = []
    references: list[str] = []
    count = 0

    with results_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(batched(dataset_split, args.batch_size), total=batch_count(len(dataset_split), args.batch_size)):
            dialogues = [str(example["dialogue"]) for example in batch]
            refs = [str(example["summary"]) for example in batch]
            prompts = [PROMPT_TEMPLATE.format(dialogue=dialogue) for dialogue in dialogues]
            preds = generate_batch(model, tokenizer, prompts, args)

            for example, prompt, pred, ref in zip(batch, prompts, preds, refs):
                sample_id = example.get("id", example.get("fname", count))
                row = {
                    "id": sample_id,
                    "split": split_name,
                    "prompt": prompt,
                    "dialogue": example["dialogue"],
                    "prediction": pred,
                    "reference": ref,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                predictions.append(pred)
                references.append(ref)
                count += 1

    rouge = compute_rouge(predictions, references)
    summary = {
        "model": args.model_name_or_path,
        "dataset": args.dataset,
        "split": split_name,
        "num_samples": count,
        "prompt_template": PROMPT_TEMPLATE,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        },
        "rouge": rouge,
        "outputs": {
            "results_jsonl": str(results_path),
            "summary_json": str(output_dir / "summary.json"),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"num_samples": count, "rouge": rouge, "output_dir": str(output_dir)}, indent=2))


def load_model(args: argparse.Namespace):
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model.to(args.device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_dialogsum(args: argparse.Namespace):
    download_config = DownloadConfig(local_files_only=args.local_files_only)
    try:
        split = load_dataset(args.dataset, split=args.split, download_config=download_config)
        return split, args.split
    except Exception:
        if args.split == "test":
            raise
        split = load_dataset(args.dataset, split="test", download_config=download_config)
        return split, "test"


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
    )
    inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_length = inputs["input_ids"].shape[1]
    outputs = []
    for tokens in generated:
        new_tokens = tokens[prompt_length:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def batched(dataset_split, batch_size: int) -> Iterable[list[dict]]:
    batch = []
    for example in dataset_split:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def batch_count(num_items: int, batch_size: int) -> int:
    return (num_items + batch_size - 1) // batch_size


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if not predictions:
        return totals

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_for_rouge(pred)
        ref_tokens = tokenize_for_rouge(ref)
        totals["rouge1"] += rouge_n_f1(pred_tokens, ref_tokens, 1)
        totals["rouge2"] += rouge_n_f1(pred_tokens, ref_tokens, 2)
        totals["rougeL"] += rouge_l_f1(pred_tokens, ref_tokens)

    return {name: value / len(predictions) for name, value in totals.items()}


def tokenize_for_rouge(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rouge_n_f1(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    pred_ngrams = ngrams(pred_tokens, n)
    ref_ngrams = ngrams(ref_tokens, n)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum((Counter(pred_ngrams) & Counter(ref_ngrams)).values())
    return f1(overlap, len(pred_ngrams), len(ref_ngrams))


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_l_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = lcs_len(pred_tokens, ref_tokens)
    return f1(overlap, len(pred_tokens), len(ref_tokens))


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def f1(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap == 0 or pred_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    main()
