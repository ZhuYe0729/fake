#!/usr/bin/env python3
"""Restartable multi-GPU dispatcher for fixed-block real-vLLM NLL labels."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import MODELS, model_root, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--gpus", default="0", help="comma-separated GPU IDs")
    parser.add_argument("--selection", help="comma-separated policy IDs; default is all 72")
    parser.add_argument("--blocks", type=int, default=100)
    return parser.parse_args()


def complete(path: Path, expected_policy_hash: str, expected_sample_hash: str, blocks: int) -> bool:
    try:
        payload = json.loads(path.read_text())
        runtime = payload["runtime"]
        return payload["label"].startswith("p") and len(payload["blocks"]) == blocks and runtime["sample_sha256"] == expected_sample_hash and ("policy_sha256" not in runtime or runtime["policy_sha256"] == expected_policy_hash)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    root = model_root(args.model)
    manifest_path = root / "policies/prefill_only/manifest.json"
    samples = root / "samples/wikitext_2048_targets.pt"
    if not manifest_path.exists() or not samples.exists():
        raise RuntimeError("run generate_inputs.py first")
    manifest = json.loads(manifest_path.read_text())
    selected = set(args.selection.split(",")) if args.selection else {row["policy_id"] for row in manifest}
    known = {row["policy_id"] for row in manifest}
    if unknown := selected - known:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    expected_samples = sha256(samples)
    jobs = [row for row in manifest if row["policy_id"] in selected and not complete(root / "results" / f"{row['policy_id']}.json", row["sha256"], expected_samples, args.blocks)]
    state_path = root / "run_state.json"
    state = {"model": args.model, "requested": sorted(selected), "blocks": args.blocks, "pending": [row["policy_id"] for row in jobs], "completed": [], "failed": []}
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    workers: dict[str, tuple[dict, subprocess.Popen]] = {}
    script = Path(__file__).with_name("stream_policy_nll.py")
    while jobs or workers:
        for gpu in gpus:
            if gpu in workers or not jobs:
                continue
            row = jobs.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            process = subprocess.Popen([sys.executable, str(script), "--model", args.model, "--policy", row["policy_id"], "--blocks", str(args.blocks)], env=env)
            workers[gpu] = (row, process)
        time.sleep(2)
        for gpu, (row, process) in list(workers.items()):
            code = process.poll()
            if code is None:
                continue
            del workers[gpu]
            result = root / "results" / f"{row['policy_id']}.json"
            if code == 0 and complete(result, row["sha256"], expected_samples, args.blocks):
                state["completed"].append(row["policy_id"])
            else:
                state["failed"].append({"policy": row["policy_id"], "gpu": gpu, "exit_code": code})
            state["pending"] = [item["policy_id"] for item in jobs] + [item[0]["policy_id"] for item in workers.values()]
            state_path.write_text(json.dumps(state, indent=2) + "\n")
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    if state["failed"]:
        raise RuntimeError(f"failed policies: {state['failed']}")
    print(json.dumps({"completed": len(state["completed"]), "skipped": len(selected) - len(state["completed"])}, indent=2))


if __name__ == "__main__":
    main()
