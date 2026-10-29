#!/usr/bin/env python
"""Standalone PMPD-style zero-shot evaluation for local causal LMs."""

from __future__ import annotations

import argparse
import html
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/pmpd_eval_hf_datasets_cache")
os.environ.setdefault("HF_HOME", "/tmp/pmpd_eval_hf_home")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pmpd_eval_matplotlib")

import torch
from datasets import Dataset, load_dataset, load_from_disk
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATA_ROOT = Path("/home/agent/wja/data/datasets/flaxquant")
DEFAULT_MODEL_PATH = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
DEFAULT_OUTPUT_DIR = Path("outputs/pmpd_style_eval")
DEFAULT_BERTSCORE_MODEL = Path("/home/agent/wja/data/models/bert_score/roberta-large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PMPD-style generation/evaluation.")
    parser.add_argument("--dataset", choices=["cnn_dm", "cnn_dm_1000", "dsum", "IWSLT"], required=True)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-id", default="llama2-7b")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=3840)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--bertscore-model",
        type=Path,
        default=DEFAULT_BERTSCORE_MODEL,
        help="Local roberta-large path used for CNN/DM and DialogSum BERTScore.",
    )
    parser.add_argument(
        "--bertscore-num-layers",
        type=int,
        default=17,
        help="BERTScore layer setting. PMPD uses roberta-large through evaluate/bertscore.",
    )
    parser.add_argument(
        "--iwslt-filter-tokenizer",
        default="lmsys/vicuna-7b-v1.5",
        help="Tokenizer used by PMPD to keep IWSLT examples with English length > 60.",
    )
    parser.add_argument(
        "--metrics-only",
        type=Path,
        default=None,
        help="Only compute metrics from an existing PMPD-style jsonl answer file.",
    )
    return parser.parse_args()


def prompt_cnn(article: str) -> str:
    return f"""

For the following article: {article} 

Return a summary comprising of around 3 sentences.

"""


def prompt_dsum(dialogue: str) -> str:
    return f"""

For the following dialogue: {dialogue} 

Return a summary comprising of 1 or 2 sentences.

"""


def prompt_translation(src: str) -> str:
    return f"Translate the following text from French to English: {src}"


def claude_prompt(user_message: str) -> str:
    # FastChat's Claude template is ADD_COLON_SINGLE with roles Human/Assistant
    # and sep="\n\n"; PMPD prepends the role-play instruction below.
    prompt = f"Human: {user_message}\n\nAssistant: "
    return "Play the role of assistant and answer the question from human. " + prompt


def load_cnn_dm(split: str, data_root: Path) -> Dataset:
    repo_dir = data_root / "cnn_dailymail_repo" / "3.0.0"
    if repo_dir.exists():
        files = sorted(repo_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No CNN/DM parquet files for split={split}: {repo_dir}")
        return load_dataset("parquet", data_files={split: [str(path) for path in files]}, split=split)
    return load_dataset("cnn_dailymail", "3.0.0", split=split)


def load_cnn_dm_subset(split: str, data_root: Path) -> Dataset:
    subset_dir = data_root / "cnn_dailymail_3.0.0_test_random1000_seed42"
    if not subset_dir.exists():
        raise FileNotFoundError(
            f"Missing fixed CNN/DM subset: {subset_dir}. "
            "Create it with make_cnn_dm_subset.py first."
        )
    dataset_dict = load_from_disk(str(subset_dir))
    return dataset_dict[split]


def load_dsum(split: str, data_root: Path) -> Dataset:
    repo_dir = data_root / "dialogsum_repo"
    csv_path = repo_dir / f"{split}.csv"
    if csv_path.exists():
        return load_dataset("csv", data_files={split: str(csv_path)}, split=split)
    return load_dataset("knkarthick/dialogsum", split=split)


def load_iwslt_raw(split: str, data_root: Path) -> list[dict[str, dict[str, str]]]:
    repo_dir = data_root / "iwslt2017_en_fr_repo"
    zip_path = repo_dir / "data/2017-01-trnted/texts/en/fr/en-fr.zip"
    if zip_path.exists():
        return load_iwslt_from_zip(zip_path, split)
    return load_dataset("IWSLT/iwslt2017", "iwslt2017-en-fr", split=split).to_list()


def load_iwslt_from_zip(zip_path: Path, split: str) -> list[dict[str, dict[str, str]]]:
    if split == "train":
        pairs = [("en-fr/train.tags.en-fr.en", "en-fr/train.tags.en-fr.fr")]
    elif split == "validation":
        pairs = [("en-fr/IWSLT17.TED.dev2010.en-fr.en.xml", "en-fr/IWSLT17.TED.dev2010.en-fr.fr.xml")]
    elif split == "test":
        pairs = [
            (
                f"en-fr/IWSLT17.TED.tst{year}.en-fr.en.xml",
                f"en-fr/IWSLT17.TED.tst{year}.en-fr.fr.xml",
            )
            for year in range(2010, 2016)
        ]
    else:
        raise KeyError(f"Unsupported IWSLT split: {split}")

    rows: list[dict[str, dict[str, str]]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for source_name, target_name in pairs:
            source_lines = read_iwslt_lines(archive, source_name)
            target_lines = read_iwslt_lines(archive, target_name)
            for source, target in zip(source_lines, target_lines):
                rows.append({"translation": {"en": source, "fr": target}})
    return rows


def read_iwslt_lines(archive: zipfile.ZipFile, member: str) -> list[str]:
    lines: list[str] = []
    with archive.open(member) as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("<seg"):
                line = line.split(">", 1)[1].split("<", 1)[0]
            elif line.startswith("<"):
                continue
            lines.append(html.unescape(line).strip())
    return lines


def build_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset in {"cnn_dm", "cnn_dm_1000"}:
        dataset = (
            load_cnn_dm_subset(args.split, args.data_root)
            if args.dataset == "cnn_dm_1000"
            else load_cnn_dm(args.split, args.data_root)
        )
        questions = [
            {
                "question_id": i,
                "prompt": claude_prompt(prompt_cnn(row["article"])),
                "reference": row["highlights"],
            }
            for i, row in enumerate(dataset)
        ]
    elif args.dataset == "dsum":
        dataset = load_dsum(args.split, args.data_root)
        questions = [
            {
                "question_id": i,
                "prompt": claude_prompt(prompt_dsum(row["dialogue"])),
                "reference": row["summary"],
            }
            for i, row in enumerate(dataset)
        ]
    elif args.dataset == "IWSLT":
        dataset = load_iwslt_raw(args.split, args.data_root)
        filter_tokenizer = AutoTokenizer.from_pretrained(args.iwslt_filter_tokenizer)
        questions = []
        for i, row in enumerate(dataset):
            translation = row["translation"]
            if len(filter_tokenizer(translation["en"]).input_ids) > 60:
                questions.append(
                    {
                        "question_id": i,
                        "prompt": claude_prompt(prompt_translation(translation["fr"])),
                        "reference": translation["en"],
                    }
                )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return questions[args.question_begin : args.question_end]


def truncate_prompt(tokenizer: Any, prompt: str, max_input_tokens: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(token_ids) <= max_input_tokens:
        return prompt
    return tokenizer.decode(token_ids[-max_input_tokens:], skip_special_tokens=True)


def load_generation_model(model_path: Path) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    return model, tokenizer


@torch.inference_mode()
def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    max_input_tokens: int,
) -> tuple[str, int, float]:
    prompt = truncate_prompt(tokenizer, prompt, max_input_tokens)
    input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()
    input_len = input_ids.shape[1]

    torch.cuda.synchronize()
    started = time.time()
    outputs = model(input_ids, use_cache=True)
    new_token = 0

    for _ in range(max_new_tokens):
        input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = model(input_id, use_cache=True, past_key_values=outputs.past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        new_token += 1
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break

    torch.cuda.synchronize()
    elapsed = time.time() - started
    output = tokenizer.decode(
        input_ids[0].tolist()[input_len:],
        spaces_between_special_tokens=False,
    )
    return output, new_token, elapsed


def answer_path(args: argparse.Namespace) -> Path:
    split_suffix = "" if args.split == "test" else f"_{args.split}"
    return args.output_dir / args.dataset / f"{args.model_id}-fp16{split_suffix}.jsonl"


def run_generation(args: argparse.Namespace) -> Path:
    questions = build_questions(args)
    out_path = answer_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(args, out_path.parent, len(questions))

    model, tokenizer = load_generation_model(args.model_path)
    run_started = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as writer:
        for index, question in enumerate(questions, start=1):
            output, new_token, wall_time = greedy_generate(
                model,
                tokenizer,
                question["prompt"],
                args.max_new_tokens,
                args.max_input_tokens,
            )
            record = {
                "question_id": question["question_id"],
                "answer_id": f"local-{question['question_id']}",
                "model_id": args.model_id,
                "choices": [
                    {
                        "index": 0,
                        "turns": [output],
                        "idxs": [new_token - 1],
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
                    f"[progress] {args.dataset}: {index}/{len(questions)} "
                    f"wall_seconds={elapsed:.2f}",
                    flush=True,
                )

    return out_path


def write_run_config(args: argparse.Namespace, output_dir: Path, num_questions: int) -> None:
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
        "bertscore_model": str(args.bertscore_model),
        "bertscore_num_layers": args.bertscore_num_layers,
        "iwslt_filter_tokenizer": str(args.iwslt_filter_tokenizer),
        "pmpd_style": {
            "batch_size": 1,
            "decoding": "greedy_argmax",
            "prompt_template": "FastChat Claude-style ADD_COLON_SINGLE plus PMPD role-play prefix",
            "iwslt_direction": "French to English",
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_answers(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_path: Path,
    num_layers: int,
) -> list[float]:
    from bert_score import score

    _, _, f1 = score(
        predictions,
        references,
        lang="en",
        model_type=str(model_path),
        num_layers=num_layers,
        verbose=False,
        rescale_with_baseline=False,
    )
    return [float(value) for value in f1]


def compute_metrics(args: argparse.Namespace, path: Path) -> Path:
    records = read_answers(path)
    raw_predictions = [record["choices"][0]["turns"][0] for record in records]
    predictions = [normalize_prediction_for_metrics(prediction) for prediction in raw_predictions]
    references = [record["reference"] for record in records]
    new_tokens = [record["choices"][0]["new_tokens"][0] for record in records]
    wall_times = [record["choices"][0]["wall_time"][0] for record in records]
    empty_predictions = sum(1 for prediction in raw_predictions if not prediction.strip())

    rouge_l = compute_rouge_l(predictions, references)
    metrics: dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(records),
        "empty_predictions": empty_predictions,
        "empty_prediction_percent": 100.0 * empty_predictions / len(records),
        "avg_new_tokens": sum(new_tokens) / len(new_tokens),
        "generation_seconds": sum(wall_times),
        "tokens_per_second": sum(new_tokens) / sum(wall_times),
        "rougeL": rouge_l,
        "rougeL_percent": rouge_l * 100,
    }

    if args.dataset in {"cnn_dm", "cnn_dm_1000", "dsum"}:
        non_empty_indices = [
            index for index, prediction in enumerate(raw_predictions) if prediction.strip()
        ]
        bert_scores = [0.0 for _ in raw_predictions]
        if non_empty_indices:
            non_empty_scores = compute_bertscore(
                [raw_predictions[index] for index in non_empty_indices],
                [references[index] for index in non_empty_indices],
                args.bertscore_model,
                args.bertscore_num_layers,
            )
            for index, score in zip(non_empty_indices, non_empty_scores):
                bert_scores[index] = score
            metrics["bert_score_non_empty"] = sum(non_empty_scores) / len(non_empty_scores)
            metrics["bert_score_non_empty_percent"] = metrics["bert_score_non_empty"] * 100
        else:
            metrics["bert_score_non_empty"] = None
            metrics["bert_score_non_empty_percent"] = None
        bert = sum(bert_scores) / len(bert_scores)
        metrics["bert_score"] = bert
        metrics["bert_score_percent"] = bert * 100
    elif args.dataset == "IWSLT":
        bleu = corpus_bleu(predictions, [references]).score
        metrics["sacre_bleu"] = bleu
        metrics["bleu_percent"] = bleu

    metrics_path = path.parent / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as writer:
        json.dump(metrics, writer, ensure_ascii=False, indent=2)
        writer.write("\n")
    print(f"[metrics] {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics_path


def normalize_prediction_for_metrics(prediction: str) -> str:
    # BERTScore cannot encode an empty sequence with some tokenizers. Treat
    # whitespace-only generations as a minimal wrong answer for Rouge/SacreBLEU
    # while separately assigning them BERTScore=0 in compute_metrics.
    return prediction if prediction.strip() else "."


def main() -> None:
    args = parse_args()
    path = args.metrics_only or run_generation(args)
    compute_metrics(args, path)


if __name__ == "__main__":
    main()
