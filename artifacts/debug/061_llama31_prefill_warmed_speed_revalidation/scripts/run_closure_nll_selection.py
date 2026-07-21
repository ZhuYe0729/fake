#!/usr/bin/env python3
"""Parallel NLL closure for the warmed-speed representative policies."""
from __future__ import annotations

import argparse
import concurrent.futures
import queue
import subprocess
import sys
from pathlib import Path

from scenario import EXP
from run_closure_speed_selection import DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="3,4,5,6")
    parser.add_argument("--policies", help="comma-separated policy IDs")
    args = parser.parse_args()
    selected = tuple(args.policies.split(",")) if args.policies else DEFAULT
    gpus = [int(item) for item in args.gpus.split(",") if item]
    pending = [policy for policy in selected if not (EXP / "pareto/closure/nll" / f"{policy}.json").exists()]
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        available.put(gpu)

    def run(policy: str) -> str:
        gpu = available.get()
        try:
            subprocess.run([sys.executable, str(Path(__file__).with_name("run_closure_nll_policy.py")),
                            "--policy", policy, "--gpu", str(gpu)], check=True)
            return policy
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for policy in pool.map(run, pending):
            print(f"completed {policy}", flush=True)


if __name__ == "__main__":
    main()
