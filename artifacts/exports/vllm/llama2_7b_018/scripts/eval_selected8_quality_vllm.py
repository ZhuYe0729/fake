#!/usr/bin/env python3
"""Evaluate selected-8 vLLM checkpoints on ARC-Challenge with lm-eval."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import lm_eval


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_DENSE_BF16_MODEL = Path("/root/wja/data/models/LLM-Research/llama-2-7b")
MODEL_SPECS = {
    "dense_bf16": DEFAULT_DENSE_BF16_MODEL,
    "dense_nvfp4": BASELINE_ROOT / "uniform_dense_nvfp4",
    "sparse_bf16": BASELINE_ROOT / "uniform_sparse_bf16",
    "sparse_nvfp4": BASELINE_ROOT / "uniform_sparse_nvfp4",
    "marlin_nvfp4": BASELINE_ROOT / "uniform_marlin_nvfp4",
    "hetero_strategy_a": BASELINE_ROOT / "hetero_strategy_a",
    "hetero_strategy_b": BASELINE_ROOT / "hetero_strategy_b",
    "hetero_strategy_c": BASELINE_ROOT / "hetero_strategy_c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASELINE_ROOT / "quality/selected_8_scenarios",
    )
    parser.add_argument(
        "--methods",
        default="hetero_strategy_a,hetero_strategy_b,hetero_strategy_c",
    )
    parser.add_argument("--task", default="arc_challenge")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_methods(args.methods)
    rows = []
    for method in methods:
        model_path = MODEL_SPECS[method]
        if not model_path.exists():
            raise FileNotFoundError(f"{method} model path does not exist: {model_path}")
        row = evaluate_method(method, model_path, args)
        rows.append(row)
        write_json(args.output_dir / f"{method}_quality.json", row)
        write_csv(args.output_dir / "selected8_vllm_quality.csv", rows)


def parse_methods(spec: str) -> list[str]:
    methods = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [method for method in methods if method not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported={sorted(MODEL_SPECS)}")
    return methods


def evaluate_method(method: str, model_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    print(f"evaluating {method}: {model_path}", flush=True)
    os.environ.setdefault("HF_HOME", "/home/agent/wja/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/agent/wja/.cache/huggingface/datasets")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_args = {
        "pretrained": str(model_path),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": not args.no_enforce_eager,
        "enable_prefix_caching": False,
        "trust_remote_code": False,
        "seed": args.seed,
    }
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=[args.task],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate returned None")
    task_result = results["results"][args.task]
    row = {
        "method": method,
        "model_path": str(model_path),
        "task": args.task,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "sample_len": task_result.get("sample_len", ""),
        "acc": first_metric(task_result, "acc"),
        "acc_norm": first_metric(task_result, "acc_norm"),
        "raw_results": task_result,
    }
    print(json.dumps(row, indent=2, default=str), flush=True)
    return row


def first_metric(result: dict[str, Any], prefix: str) -> float | None:
    for key, value in result.items():
        if key.startswith(prefix + ","):
            return float(value)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat.pop("raw_results", None)
        flat_rows.append(flat)
    if not flat_rows:
        return
    fields = list(flat_rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


if __name__ == "__main__":
    main()
