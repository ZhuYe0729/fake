#!/usr/bin/env python3
"""Run the paper-facing prefill task selection in parallel, one policy per GPU."""
from __future__ import annotations

import argparse
import concurrent.futures
import queue
import subprocess
import sys
from pathlib import Path

from scenario import EXP

SELECTED = (
    "p00", "p01", "p02", "p03", "p04",
    "point_003", "point_005", "point_007",
    "bridge_dense_nvfp4_072", "bridge_dense_nvfp4_088",
    "bridge_dense_nvfp4_104", "bridge_dense_nvfp4_120",
    "point_009", "point_011", "point_014",
)
TASKS = ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu")


def policy_path(label: str) -> Path:
    root = EXP / "policies/prefill_only" if len(label) == 3 and label.startswith("p") and label[1:].isdigit() else EXP / "pareto/policies"
    path = root / f"{label}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def complete(label: str) -> bool:
    root = EXP / "task_quality/results" / label
    return all((root / task / "full/result.json").exists() for task in TASKS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--labels", help="Comma-separated labels; defaults to paper closure selection.")
    args = parser.parse_args()
    selected = tuple(args.labels.split(",")) if args.labels else SELECTED
    pending = [label for label in selected if not complete(label)]
    if not pending:
        print("all selected task results already exist")
        return
    gpus = [int(value) for value in args.gpus.split(",") if value]
    if not gpus:
        raise ValueError("at least one GPU is required")
    logs = EXP / "logs/tasks"
    logs.mkdir(parents=True, exist_ok=True)
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        available.put(gpu)

    def run(label: str) -> tuple[str, str | None]:
        gpu = available.get()
        try:
            try:
                command = [sys.executable, str(Path(__file__).with_name("run_task_policy.py")),
                           "--policy-json", str(policy_path(label)), "--label", label, "--gpu", str(gpu)]
                with (logs / f"{label}.log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                return label, None if completed.returncode == 0 else f"exit={completed.returncode}"
            except Exception as error:
                return label, f"{type(error).__name__}: {error}"
        finally:
            available.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for label, error in pool.map(run, pending):
            print(f"completed {label}" if error is None else f"failed {label}: {error}", flush=True)


if __name__ == "__main__":
    main()
