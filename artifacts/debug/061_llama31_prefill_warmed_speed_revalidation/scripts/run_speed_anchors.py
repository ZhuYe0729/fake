#!/usr/bin/env python3
"""Dispatch at most one warmed speed policy per requested GPU."""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import json
import os
import queue
import subprocess
import sys
from pathlib import Path
from scenario import EXP


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--gpus", default="1,2,3,4"); args = parser.parse_args()
    gpus = [int(item) for item in args.gpus.split(",") if item]
    policies = [row["policy_id"] for row in csv.DictReader((EXP / "speed/calibration/design.csv").open())]
    pending = [item for item in policies if not (EXP / "speed/calibration/runs" / f"{item}.json").exists()]
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus: available.put(gpu)
    logs = EXP / "logs"; logs.mkdir(parents=True, exist_ok=True)
    state = {"queued": len(pending), "completed": [], "failed": []}
    state_path = EXP / "speed/calibration/run_state.json"; state_path.write_text(json.dumps(state, indent=2) + "\n")
    def run(policy: str) -> tuple[str, int]:
        gpu = available.get()
        try:
            with (logs / f"{policy}_gpu{gpu}.log").open("w") as log:
                result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_speed_policy.py")), "--policy", policy, "--gpu", str(gpu)], stdout=log, stderr=subprocess.STDOUT)
            return policy, result.returncode
        finally:
            available.put(gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for policy, code in pool.map(run, pending):
            (state["completed"] if code == 0 else state["failed"]).append(policy)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
    if state["failed"]: raise SystemExit(json.dumps(state["failed"]))


if __name__ == "__main__": main()
