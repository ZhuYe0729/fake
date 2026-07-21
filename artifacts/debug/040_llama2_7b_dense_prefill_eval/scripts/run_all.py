#!/usr/bin/env python3
"""Schedule the five dense lm-eval tasks on exactly four independent GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("evaluate_task.py")
TASKS = ("wikitext", "c4", "winogrande", "arc_easy", "mmlu")
PPL_TASKS = {"wikitext", "c4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3", help="Exactly four physical GPU ids, comma-separated")
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--ppl-batch-size", default="1", help="Batch size for rolling-PPL tasks with long documents")
    parser.add_argument("--limit", type=int, help="Use only for smoke runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy", action="store_true", help="Use http://127.0.0.1:8848 and socks5://127.0.0.1:8848 for dataset downloads")
    parser.add_argument("--force", action="store_true", help="Re-run tasks with an existing result")
    return parser.parse_args()


def profile(limit: int | None) -> str:
    return "full" if limit is None else f"limit_{limit}"


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(item.isdigit() for item in gpus):
        raise ValueError("--gpus must contain exactly four distinct non-negative GPU ids")
    return gpus


def result_path(task: str, run_profile: str) -> Path:
    return ROOT / "tasks" / task / run_profile / "result.json"


def task_batch_size(task: str, args: argparse.Namespace) -> str:
    return args.ppl_batch_size if task in PPL_TASKS else args.batch_size


def launch(task: str, gpu: str, args: argparse.Namespace, run_profile: str) -> tuple[subprocess.Popen[str], object, Path]:
    output = result_path(task, run_profile)
    log = output.with_name("stdout_stderr.log")
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["TOKENIZERS_PARALLELISM"] = "false"
    if args.proxy:
        env.update({
            "http_proxy": "http://127.0.0.1:8848",
            "https_proxy": "http://127.0.0.1:8848",
            "all_proxy": "socks5://127.0.0.1:8848",
            "HF_HUB_DISABLE_XET": "1",
        })
    command = [sys.executable, str(RUNNER), "--task", task, "--output", str(output), "--batch-size", task_batch_size(task, args), "--seed", str(args.seed)]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    (output.parent / "command.json").write_text(json.dumps({"physical_gpu": gpu, "command": command, "proxy_enabled": args.proxy}, indent=2) + "\n")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
    return process, handle, output


def main() -> None:
    args = parse_args()
    gpus = parse_gpus(args.gpus)
    run_profile = profile(args.limit)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    pending = [task for task in TASKS if args.force or not result_path(task, run_profile).exists()]
    completed = [task for task in TASKS if task not in pending]
    active: dict[str, tuple[str, subprocess.Popen[str], object, Path]] = {}
    failures: dict[str, int] = {}

    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            task = pending.pop(0)
            process, handle, output = launch(task, gpu, args, run_profile)
            active[gpu] = (task, process, handle, output)
            print(f"launched task={task} gpu={gpu} pid={process.pid}", flush=True)
        time.sleep(1)
        for gpu, (task, process, handle, output) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[gpu]
            if code == 0 and output.exists():
                completed.append(task)
                print(f"completed task={task} gpu={gpu}", flush=True)
            else:
                failures[task] = code
                print(f"failed task={task} gpu={gpu} returncode={code}", flush=True)

    elapsed = time.perf_counter() - started
    summary = {
        "profile": run_profile,
        "gpus": gpus,
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
        "completed": completed,
        "failures": failures,
    }
    summary_path = ROOT / "runs" / run_profile / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if failures:
        raise SystemExit(f"failed tasks: {failures}")


if __name__ == "__main__":
    main()
