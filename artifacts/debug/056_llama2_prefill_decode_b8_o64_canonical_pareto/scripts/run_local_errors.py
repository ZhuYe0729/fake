#!/usr/bin/env python3
"""Run one canonical local-error job per available GPU, resumably."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
import sys
from pathlib import Path

from scenario import EXP

METHODS = ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    parser.add_argument("--blocks", type=int, default=16)
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value]
    jobs = [(phase, method) for phase in ("prefill", "decode") for method in METHODS]
    jobs = [job for job in jobs if not (EXP / "local_errors" / f"{job[0]}_{job[1]}.csv").exists()]
    logs = EXP.parent.parent / "logs/local_errors"; logs.mkdir(parents=True, exist_ok=True)

    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        available.put(gpu)

    def run(job: tuple[str, str]) -> tuple[str, str]:
        phase, method = job
        gpu = available.get()
        command = [sys.executable, str(Path(__file__).with_name("collect_canonical_phase_local_errors.py")),
                   "--phase", phase, "--method", method, "--gpu", str(gpu), "--blocks", str(args.blocks)]
        try:
            with (logs / f"{phase}_{method}_gpu{gpu}.log").open("w") as handle:
                subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT,
                               env=os.environ.copy())
            return phase, method
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for phase, method in pool.map(run, jobs):
            print(f"completed {phase}/{method}", flush=True)


if __name__ == "__main__":
    main()
