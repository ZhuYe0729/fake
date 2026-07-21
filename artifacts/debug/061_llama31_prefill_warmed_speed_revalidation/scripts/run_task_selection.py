#!/usr/bin/env python3
"""Parallel downstream-task closure for all representative solved policies."""
from __future__ import annotations

import argparse
import concurrent.futures
import queue
import subprocess
import sys
from pathlib import Path

from scenario import EXP
from run_closure_speed_selection import DEFAULT

TASKS = ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    parser.add_argument("--policies", help="comma-separated policy IDs")
    args = parser.parse_args()
    selected = tuple(args.policies.split(",")) if args.policies else DEFAULT
    pending = [policy for policy in selected if not all((EXP / "task_quality/results" / policy / task / "full/result.json").exists() for task in TASKS)]
    available: queue.Queue[int] = queue.Queue()
    for item in args.gpus.split(","):
        if item:
            available.put(int(item))

    def run(policy: str) -> tuple[str, int]:
        gpu = available.get()
        try:
            return policy, subprocess.run([sys.executable, str(Path(__file__).with_name("run_task_policy.py")),
                                           "--policy", policy, "--gpu", str(gpu)]).returncode
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=available.qsize()) as pool:
        for policy, code in pool.map(run, pending):
            print(f"completed {policy}" if code == 0 else f"failed {policy}: exit={code}", flush=True)


if __name__ == "__main__":
    main()
