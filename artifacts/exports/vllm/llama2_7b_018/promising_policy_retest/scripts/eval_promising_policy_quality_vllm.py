#!/usr/bin/env python3
"""Evaluate promising optimized-policy vLLM checkpoints on ARC-Challenge."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import lm_eval


SCRIPT_DIR = Path(__file__).resolve().parent
RETEST_ROOT = SCRIPT_DIR.parent
DEFAULT_POLICY_DIR = RETEST_ROOT / "policies/unique_policies"
DEFAULT_CHECKPOINT_DIR = RETEST_ROOT / "checkpoints"
DEFAULT_OUTPUT_DIR = RETEST_ROOT / "quality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policies", default="")
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
    policy_names = parse_policies(args)
    rows = []
    for policy_name in policy_names:
        model_path = args.checkpoint_dir / policy_name
        if not model_path.exists():
            raise FileNotFoundError(f"policy checkpoint does not exist: {model_path}")
        row = evaluate_policy(policy_name, model_path, args)
        rows.append(row)
        write_json(args.output_dir / f"{policy_name}_quality.json", row)
        write_csv(args.output_dir / "optimized_policy_quality.csv", rows)


def parse_policies(args: argparse.Namespace) -> list[str]:
    if args.policies.strip():
        return [item.strip() for item in args.policies.split(",") if item.strip()]
    policies = []
    for path in sorted(args.policy_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        policies.append(str(payload["policy_name"]))
    return policies


def evaluate_policy(policy_name: str, model_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    print(f"evaluating {policy_name}: {model_path}", flush=True)
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
        "policy_name": policy_name,
        "model_path": str(model_path),
        "task": args.task,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "sample_len": task_result.get("sample_len", ""),
        "arc_acc": first_metric(task_result, "acc"),
        "arc_acc_norm": first_metric(task_result, "acc_norm"),
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
