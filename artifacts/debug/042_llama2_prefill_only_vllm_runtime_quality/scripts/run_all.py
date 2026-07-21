#!/usr/bin/env python3
"""Run one real-vLLM process per Llama2 prefill-only policy."""
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
TASKS = ("wikitext", "winogrande", "arc_easy", "arc_challenge", "mmlu")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--selection", help="comma-separated policy labels")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def complete(root: Path, label: str, profile: str) -> bool:
    try:
        return all(json.loads((root / label / task / profile / "result.json").read_text()).get("metrics") for task in TASKS)
    except (OSError, ValueError):
        return False


def main() -> None:
    a = args()
    if not MANIFEST.exists():
        subprocess.run([sys.executable, str(DEBUG / "scripts/build_manifest.py")], check=True)
    manifest = json.loads(MANIFEST.read_text())
    wanted = set(a.selection.split(",")) if a.selection else None
    policies = [item for item in manifest["policies"] if wanted is None or item["label"] in wanted]
    if wanted and wanted != {item["label"] for item in policies}:
        raise ValueError(f"unknown policy: {sorted(wanted - {item['label'] for item in policies})}")
    profile = "full" if a.limit is None else f"limit_{a.limit}"
    jobs = [item for item in policies if a.force or not complete(DEBUG / "results", item["label"], profile)]
    gpus = [x.strip() for x in a.gpus.split(",") if x.strip()]
    state_path = DEBUG / "run_state" / f"{profile}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"profile": profile, "queued": len(jobs), "completed": 0, "failed": [], "gpus": gpus}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    logs = DEBUG / "logs" / profile; logs.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, dict, object]] = {}
    while jobs or running:
        for gpu in gpus:
            if gpu in running or not jobs:
                continue
            item = jobs.pop(0)
            cmd = [sys.executable, str(DEBUG / "scripts/evaluate_policy.py"), "--policy", item["label"]]
            if a.limit is not None:
                cmd += ["--limit", str(a.limit)]
            if a.audit:
                cmd.append("--audit")
            env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false")
            handle = (logs / f"{item['label']}.log").open("w")
            process = subprocess.Popen(cmd, cwd=DEBUG, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (process, item, handle)
            print(f"started gpu={gpu} {item['label']}", flush=True)
        time.sleep(5)
        for gpu, (process, item, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close(); del running[gpu]; state["completed"] += 1
            if code != 0 or not complete(DEBUG / "results", item["label"], profile):
                state["failed"].append({"gpu": gpu, "policy": item["label"], "exit_code": code})
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"finished gpu={gpu} code={code} {item['label']}", flush=True)
    if state["failed"]:
        raise SystemExit(f"failed policies: {state['failed']}")


if __name__ == "__main__":
    main()
