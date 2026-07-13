#!/usr/bin/env python3
"""Merge isolated PMPD shard JSONL files, then calculate one metric report."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--python", default="/home/agent/wja/miniconda3/envs/vllm/bin/python")
    args = parser.parse_args()
    records = {}
    for path in sorted(args.shard_root.glob("shard_*/**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            key = str(item["question_id"])
            if key in records:
                raise RuntimeError(f"duplicate question_id={key} from {path}")
            records[key] = item
    target = args.output_dir / args.dataset / f"{args.label}-fp16.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda item: int(item["question_id"]) if str(item["question_id"]).isdigit() else str(item["question_id"]))
    target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")
    evaluator = REPO_ROOT / "references/pmpd_eval_kit/pmpd_eval.py"
    subprocess.run([args.python, str(evaluator), "--dataset", args.dataset, "--metrics-only", str(target)], check=True)
    print(json.dumps({"records": len(ordered), "jsonl": str(target), "metrics": str(target.with_name("metrics.json"))}, indent=2))


if __name__ == "__main__":
    main()
