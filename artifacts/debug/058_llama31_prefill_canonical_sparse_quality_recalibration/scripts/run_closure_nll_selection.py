#!/usr/bin/env python3
"""Parallel NLL closure for solved points; speed is intentionally excluded."""
from __future__ import annotations
import argparse
import concurrent.futures
import queue
import subprocess
import sys
from pathlib import Path
from run_closure_selection import SELECTED
from scenario import EXP

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    parser.add_argument("--policies", help="Comma-separated policy ids; defaults to the solved closure selection.")
    args = parser.parse_args()
    selected = tuple(args.policies.split(",")) if args.policies else SELECTED
    gpus = [int(x) for x in args.gpus.split(",") if x]; pending = [p for p in selected if not (EXP / "pareto/closure/nll" / f"{p}.json").exists()]
    available: queue.Queue[int] = queue.Queue(); [available.put(gpu) for gpu in gpus]
    def run(policy: str) -> str:
        gpu = available.get()
        try:
            subprocess.run([sys.executable, str(Path(__file__).with_name("run_closure_nll_point.py")), "--policy", policy, "--gpu", str(gpu)], check=True)
            return policy
        finally: available.put(gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for policy in pool.map(run, pending): print(f"completed {policy}", flush=True)
if __name__ == "__main__": main()
