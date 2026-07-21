#!/usr/bin/env python3
"""Run the frozen pure-prefill NLL design, one policy process per GPU at once."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import queue
import subprocess
import sys
from pathlib import Path
from scenario import EXP

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--gpus", default="1,2,3,4,5,6,7"); args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",") if value]
    manifest = json.loads((EXP / "policies/prefill_only/manifest.json").read_text())
    pending = [row["policy_id"] for row in manifest if not (EXP / "nll/raw" / f"{row['policy_id']}.json").exists()]
    if not pending: return
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus: available.put(gpu)
    logs = EXP / "logs/nll_scheduler"; logs.mkdir(parents=True, exist_ok=True)
    state_path = EXP / "nll/run_state.json"; state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"queued": len(pending), "completed": [], "failed": []}; state_path.write_text(json.dumps(state, indent=2) + "\n")
    def run(policy: str) -> tuple[str, int]:
        gpu = available.get()
        try:
            with (logs / f"{policy}_gpu{gpu}.log").open("w") as log:
                result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_nll_policy.py")), "--policy", policy, "--gpu", str(gpu)], stdout=log, stderr=subprocess.STDOUT)
            return policy, result.returncode
        finally:
            available.put(gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for policy, code in pool.map(run, pending):
            (state["completed"] if code == 0 else state["failed"]).append({"policy": policy, "exit_code": code})
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"{'completed' if code == 0 else 'failed'} {policy}", flush=True)
    if state["failed"]: raise SystemExit(json.dumps(state["failed"]))
if __name__ == "__main__": main()
