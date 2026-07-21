#!/usr/bin/env python3
"""Run the fixed B=8/O=64 speed-calibration design without GPU contention."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import queue
import shutil
import subprocess
from pathlib import Path

from scenario import EXP


def complete(policy_id: str) -> bool:
    path = EXP / "speed/runs" / policy_id / "iterations.csv"
    if not path.exists():
        return False
    rows = list(csv.DictReader(path.open()))
    return sum(row["phase"] == "main" and row["warmup"] == "False" for row in rows) == 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    args = parser.parse_args()
    design = list(csv.DictReader((EXP / "speed/calibration/design.csv").open()))
    todo = [row for row in design if not complete(row["policy_id"])]
    gpus = [item for item in args.gpus.split(",") if item]
    available: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        available.put(gpu)
    logs = EXP.parent.parent / "logs/speed_calibration"; logs.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_speed_policy.sh")

    def run(row: dict[str, str]) -> str:
        gpu = available.get()
        try:
            with (logs / f"{row['policy_id']}_gpu{gpu}.log").open("w") as handle:
                subprocess.run(["bash", str(runner), row["policy_id"], row["policy_json"], gpu],
                               check=True, stdout=handle, stderr=subprocess.STDOUT,
                               env=os.environ.copy())
            # Fresh-process summaries are the retained speed evidence.  The
            # materialized checkpoint is deterministic from the policy and
            # canonical states and would otherwise consume one model per row.
            shutil.rmtree(EXP / "speed/runs" / row["policy_id"] / "checkpoint",
                          ignore_errors=True)
            return row["policy_id"]
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for policy_id in pool.map(run, todo):
            print(f"completed {policy_id}", flush=True)


if __name__ == "__main__":
    main()
