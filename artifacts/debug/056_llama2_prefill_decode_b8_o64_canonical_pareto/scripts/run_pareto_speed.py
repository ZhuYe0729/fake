#!/usr/bin/env python3
"""Run resumable fresh-process speed closure for solved Pareto policies."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import time
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "llama2_7b_chat"))
RUNNER = Path(os.environ.get("COSPAQ_SPEED_RUNNER", ROOT / "scripts/run_speed_policy.sh"))


def complete(policy_id: str) -> bool:
    path = EXP / "speed/runs" / policy_id / "summary.csv"
    if not path.exists():
        return False
    try:
        return next(csv.DictReader(path.open())).get("status") == "OK"
    except (StopIteration, OSError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--policies", default="")
    parser.add_argument("--startup-stagger-seconds", type=float, default=12.0)
    args = parser.parse_args()
    pareto = EXP / "pareto"
    predicted_path = pareto / "candidates/predicted_points.csv"
    if not predicted_path.exists():
        predicted_path = pareto / "predicted_points.csv"
    policy_dir = pareto / "candidates/policies"
    if not policy_dir.exists():
        policy_dir = pareto / "policies"
    predicted = list(csv.DictReader(predicted_path.open()))
    selected = {item.strip() for item in args.policies.split(",") if item.strip()}
    policy_ids = [row["policy_id"] for row in predicted if not selected or row["policy_id"] in selected]
    todo = [policy_id for policy_id in policy_ids if not complete(policy_id)]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    logs = EXP / "speed/pareto_logs"; logs.mkdir(parents=True, exist_ok=True)

    def run(index: int, policy_id: str) -> str:
        if index < len(gpus) and index:
            time.sleep(index * args.startup_stagger_seconds)
        policy = policy_dir / f"{policy_id}.json"
        env = os.environ.copy()
        with (logs / f"{policy_id}.log").open("w") as handle:
            subprocess.run(["bash", str(RUNNER), policy_id, str(policy), gpus[index % len(gpus)]],
                           cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        # The speed summary is the retained evidence.  Policies and canonical
        # states deterministically recreate the export if a downstream task
        # needs it, so do not retain one 15+ GiB checkpoint per Pareto point.
        shutil.rmtree(EXP / "speed/runs" / policy_id / "checkpoint", ignore_errors=True)
        return policy_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run, index, policy_id) for index, policy_id in enumerate(todo)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)
    print(json.dumps({"completed": len(policy_ids), "new": len(todo)}))


if __name__ == "__main__":
    main()
