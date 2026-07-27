#!/usr/bin/env python3
"""Run selected full tasks in a restartable multi-GPU queue."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import RUN, gpu_list, runtime_env, write_json


def policy_for(label: str) -> Path:
    if label.startswith("uniform_p"):
        return RUN / "policies/prefill_only" / f"{label.removeprefix('uniform_')}.json"
    return RUN / "pareto/policies" / f"{label}.json"


def complete(label: str) -> bool:
    base = RUN / "pareto/validation/tasks" / label
    return all((base / task / "full/result.json").is_file() for task in
               ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=",".join(gpu_list()))
    parser.add_argument("--selection")
    args = parser.parse_args()
    frozen = json.loads((RUN / "tasks/selection.json").read_text())["selected"]
    selected = args.selection.split(",") if args.selection else frozen
    jobs = [label for label in selected if not complete(label)]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    workers: dict[str, tuple[str, subprocess.Popen]] = {}
    state = {"selected": selected, "completed": [], "failed": []}
    while jobs or workers:
        for gpu in gpus:
            if gpu not in workers and jobs:
                label = jobs.pop(0)
                env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = gpu
                command = [sys.executable, str(Path(__file__).with_name("evaluate_tasks.py")),
                           "--policy-json", str(policy_for(label)), "--label", label,
                           "--tasks", "wikitext,winogrande,arc_easy,arc_challenge,mmlu",
                           "--experiment-root", str(RUN),
                           "--canonical-sparse-bf16-state", str(RUN / "canonical/prepared/sparse_bf16/model.pt"),
                           "--canonical-sparse-nvfp4-state", str(RUN / "canonical/prepared/sparse_nvfp4/model.pt"),
                           "--temporary-root", str(RUN / "temporary/tasks")]
                workers[gpu] = (label, subprocess.Popen(command, env=env))
        time.sleep(5)
        for gpu, (label, process) in list(workers.items()):
            if process.poll() is None:
                continue
            del workers[gpu]
            if process.returncode == 0 and complete(label):
                state["completed"].append(label)
            else:
                state["failed"].append({"label": label, "gpu": gpu, "exit_code": process.returncode})
            state["pending"] = jobs + [item[0] for item in workers.values()]
            write_json(RUN / "tasks/run_state.json", state)
    if state["failed"]:
        raise RuntimeError(f"failed task policies: {state['failed']}")


if __name__ == "__main__":
    main()
