#!/usr/bin/env python3
"""Run one dense Llama2 lm-eval task through the 037-aligned HFLM backend."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


MODEL = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
TASKS = ("wikitext", "c4", "winogrande", "arc_easy", "mmlu")
LOCAL_TASKS = Path(__file__).resolve().parents[1] / "lm_eval_tasks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def metric_values(payload: dict[str, Any], task: str) -> dict[str, Any]:
    results = payload.get("results", {})
    groups = payload.get("groups", {})
    selected = results.get(task) or groups.get(task) or {}
    return {
        key: value
        for key, value in selected.items()
        if key.startswith(("acc,", "acc_norm,", "word_perplexity,", "byte_perplexity,", "bits_per_byte,"))
    }


def main() -> None:
    args = parse_args()
    lm_eval_task = "c4_validation_only" if args.task == "c4" else args.task
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    if not MODEL.exists():
        raise FileNotFoundError(f"model is missing: {MODEL}")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    torch.cuda.set_device(0)
    device = "cuda:0"
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
        ).to(device).eval()
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager

        lm = HFLM(
            pretrained=model,
            tokenizer=str(MODEL),
            backend="causal",
            dtype=torch.bfloat16,
            device=device,
            batch_size=args.batch_size,
            trust_remote_code=False,
        )
        result = lm_eval.simple_evaluate(
            model=lm,
            tasks=[lm_eval_task],
            num_fewshot=0,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=False,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
            task_manager=TaskManager(include_path=LOCAL_TASKS),
        )
        if result is None:
            raise RuntimeError("lm_eval.simple_evaluate returned None")
        finished_at = datetime.now(timezone.utc).isoformat()
        elapsed = time.perf_counter() - started
        row = {
            "task": args.task,
            "lm_eval_task": lm_eval_task,
            "model_path": str(MODEL),
            "backend": "lm_eval.HFLM/transformers",
            "dtype": "bfloat16",
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "metrics": metric_values(result, lm_eval_task),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "lm_eval": package_version("lm-eval"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "raw_lm_eval": result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps({key: row[key] for key in ("task", "elapsed_seconds", "metrics")}, default=str), flush=True)
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
