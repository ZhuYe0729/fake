#!/usr/bin/env python3
"""Run selected solved Pareto points across a bounded number of GPUs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", default="0,8,12,15,18,20,22,23")
    parser.add_argument("--gpus", default="1,2,3,4")
    args = parser.parse_args()
    points = [point.strip() for point in args.points.split(",") if point.strip()]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    script = __file__.replace("run_pareto_validation.py", "validate_pareto_point.py")
    root = Path(__file__).resolve().parents[1] / "llama2_7b_chat/pareto/validation/logs"
    root.mkdir(parents=True, exist_ok=True)
    jobs = list(points)
    active: dict[str, tuple[str, subprocess.Popen]] = {}
    failed = []
    while jobs or active:
        for gpu in gpus:
            if gpu not in active and jobs:
                point = jobs.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                log = (root / f"point_{int(point):03d}_gpu{gpu}.log").open("w")
                process = subprocess.Popen([sys.executable, script, "--point", point], env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (point, process, log)
        time.sleep(2)
        for gpu, (point, process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[gpu]
            log.close()
            if code:
                failed.append({"point": point, "gpu": gpu, "exit_code": code})
            else:
                print(f"complete point_{int(point):03d}", flush=True)
    if failed:
        raise RuntimeError(f"failed points: {failed}")


if __name__ == "__main__":
    main()
