#!/usr/bin/env python3
"""Schedule independent policy/task evaluations without touching GPU 0."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEBUG = Path(__file__).resolve().parents[1]
MANIFEST = DEBUG / "manifest/policies.json"
TASKS = ("wikitext", "winogrande", "arc_easy", "mmlu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7", help="physical GPU IDs; GPU 0 is deliberately not the default")
    parser.add_argument("--selection", help="comma-separated model:policy entries; default is all manifest policies")
    parser.add_argument("--limit", type=int, help="diagnostic lm-eval limit; omitted means full evaluation")
    parser.add_argument("--force", action="store_true", help="rerun even if result.json already exists")
    return parser.parse_args()


def valid_result(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        return bool(payload.get("metrics"))
    except (OSError, ValueError, TypeError):
        return False


def main() -> None:
    args = parse_args()
    if not MANIFEST.exists():
        subprocess.run([sys.executable, str(DEBUG / "scripts/build_manifest.py")], check=True)
    manifest = json.loads(MANIFEST.read_text())
    selection = set(args.selection.split(",")) if args.selection else None
    policies = [p for p in manifest["policies"] if selection is None or f"{p['model']}:{p['label']}" in selection]
    unknown = selection - {f"{p['model']}:{p['label']}" for p in policies} if selection else set()
    if unknown:
        raise ValueError(f"unknown selections: {sorted(unknown)}")
    profile = "full" if args.limit is None else f"limit_{args.limit}"
    jobs: list[tuple[dict, str, Path]] = []
    for policy in policies:
        for task in TASKS:
            output = DEBUG / "results" / policy["model"] / policy["label"] / task / profile / "result.json"
            if args.force or not valid_result(output):
                jobs.append((policy, task, output))
    physical = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not physical:
        raise ValueError("at least one GPU must be selected")
    (DEBUG / "logs" / profile).mkdir(parents=True, exist_ok=True)
    (DEBUG / "run_state").mkdir(parents=True, exist_ok=True)
    state = {"profile": profile, "queued": len(jobs), "completed": 0, "failed": [], "gpus": physical, "selection": sorted(selection) if selection else "all"}
    (DEBUG / "run_state" / f"{profile}.json").write_text(json.dumps(state, indent=2) + "\n")
    running: dict[str, tuple[subprocess.Popen, dict, str, Path, object]] = {}
    while jobs or running:
        for gpu in physical:
            if gpu in running or not jobs:
                continue
            policy, task, output = jobs.pop(0)
            log = DEBUG / "logs" / profile / f"{policy['model']}__{policy['label']}__{task}.log"
            cmd = [sys.executable, str(DEBUG / "scripts/evaluate_task.py"), "--model", policy["model"], "--policy", policy["label"],
                   "--task", task, "--output", str(output), "--batch-size", "1" if task == "wikitext" else "4", "--gpu", "0"]
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            output.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false")
            handle = log.open("w")
            process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=DEBUG)
            running[gpu] = (process, policy, task, output, handle)
            print(f"started gpu={gpu} {policy['model']}:{policy['label']} {task}", flush=True)
        time.sleep(5)
        for gpu, (process, policy, task, output, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[gpu]
            state["completed"] += 1
            if code != 0 or not valid_result(output):
                state["failed"].append({"gpu": gpu, "model": policy["model"], "policy": policy["label"], "task": task, "exit_code": code})
            (DEBUG / "run_state" / f"{profile}.json").write_text(json.dumps(state, indent=2) + "\n")
            print(f"finished gpu={gpu} code={code} {policy['model']}:{policy['label']} {task}", flush=True)
    if state["failed"]:
        raise SystemExit(f"{len(state['failed'])} job(s) failed; see {DEBUG / 'run_state' / f'{profile}.json'}")


if __name__ == "__main__":
    main()
