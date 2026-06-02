#!/usr/bin/env python3
"""Quick accuracy evaluation for Qwen3.5 models using lm-evaluation-harness."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lm_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.models.qwen3_5 import QWEN3_5_VARIANTS, qwen3_5_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 accuracy via lm-eval")
    parser.add_argument("--variant", default="0.8B", choices=QWEN3_5_VARIANTS)
    parser.add_argument("--tasks", default="arc_easy", help="Comma-separated task names")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task")
    args = parser.parse_args()

    model_path = qwen3_5_model_path(args.variant)
    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"Model: {model_path}")
    print(f"Tasks: {tasks}")
    print(f"Device: {args.device}, Few-shot: {args.num_fewshot}, Batch size: {args.batch_size}")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    print(f"\n{'='*60}")
    print("Results:")
    for task in tasks:
        r = results["results"].get(task, {})
        for metric, value in r.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {task}/{metric}: {value:.4f}")
                else:
                    print(f"  {task}/{metric}: {value}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(model_path),
            "variant": args.variant,
            "tasks": tasks,
            "num_fewshot": args.num_fewshot,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results["results"],
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
