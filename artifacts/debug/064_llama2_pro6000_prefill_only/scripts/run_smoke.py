#!/usr/bin/env python3
"""Two-GPU, non-paper closure smoke for six representative policies."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import RUN, gpu_list, runtime_env, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=",".join(gpu_list()))
    args = parser.parse_args()
    labels = ("p00", "p01", "p02", "p03", "p04", "p71")
    jobs = list(labels)
    gpus = [value for value in args.gpus.split(",") if value]
    workers = {}
    state = {"diagnostic_only": True, "blocks": 2, "warmups": 1, "measured_runs": 2,
             "completed": [], "failed": []}
    while jobs or workers:
        for gpu in gpus:
            if gpu not in workers and jobs:
                label = jobs.pop(0)
                env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = gpu
                command = [sys.executable, str(Path(__file__).with_name("closure_policy.py")),
                           "--policy", str(RUN / f"policies/prefill_only/{label}.json"),
                           "--label", label, "--gpu", gpu, "--blocks", "2", "--runs", "2",
                           "--output-root", str(RUN / "smoke")]
                workers[gpu] = (label, subprocess.Popen(command, env=env))
        time.sleep(2)
        for gpu, (label, process) in list(workers.items()):
            if process.poll() is None:
                continue
            del workers[gpu]
            if process.returncode == 0:
                state["completed"].append(label)
            else:
                state["failed"].append({"label": label, "gpu": gpu, "exit_code": process.returncode})
            write_json(RUN / "smoke/state.json", state)
    if state["failed"]:
        raise RuntimeError(f"smoke failed: {state['failed']}")


if __name__ == "__main__":
    main()
