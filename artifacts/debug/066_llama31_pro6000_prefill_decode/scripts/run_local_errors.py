#!/usr/bin/env python3
"""Run all eight phase/method local-error jobs, one job per GPU at a time."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import subprocess
import sys
from pathlib import Path

from common import RUN, runtime_env

METHODS = ("dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")


def complete(path: Path, blocks: int) -> bool:
    if not path.is_file():
        return False
    import csv
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == 16 and all(int(row["blocks"]) == blocks for row in rows) and {(int(r["layer_bucket"]), r["fused_type"]) for r in rows} == {
        (bucket, typ) for bucket in range(4)
        for typ in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--blocks", type=int, default=16)
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    jobs = [(phase, method) for phase in ("prefill", "decode") for method in METHODS
            if not complete(RUN / f"local_errors/{phase}_{method}.csv", args.blocks)]
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        available.put(gpu)

    def run(job: tuple[str, str]) -> tuple[str, str]:
        phase, method = job
        gpu = available.get()
        log = RUN / f"logs/local_errors/{phase}_{method}_gpu{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).with_name("collect_phase_local_errors.py")),
                   "--phase", phase, "--method", method, "--gpu", str(gpu),
                   "--blocks", str(args.blocks)]
        try:
            with log.open("w") as handle:
                subprocess.run(command, check=True, env=runtime_env(), stdout=handle,
                               stderr=subprocess.STDOUT)
            if not complete(RUN / f"local_errors/{phase}_{method}.csv", args.blocks):
                raise RuntimeError(f"incomplete local-error output: {phase}/{method}")
            return job
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for phase, method in pool.map(run, jobs):
            print(f"completed {phase}/{method}", flush=True)


if __name__ == "__main__":
    main()
