#!/usr/bin/env python3
"""Restartable multi-GPU real-vLLM NLL runner for the 18 mechanism policies."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import CUTLASS, DEBUG, MODEL, SOURCE, VLLM_ROOT, normalized_policy, policy_methods, sha256


def complete(path: Path, policy_hash: str, sample_hash: str, blocks: int) -> bool:
    try:
        row = json.loads(path.read_text())
        return len(row["blocks"]) == blocks and row["runtime"]["sample_sha256"] == sample_hash and row["runtime"]["policy_sha256"] == policy_hash
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def one(policy_row: dict, blocks: int) -> None:
    policy_path = Path(policy_row["path"])
    output = DEBUG / "results" / f"{policy_row['policy_id']}.json"
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(os.environ.get("COSPAQ_PHASE_TMP_ROOT", "/tmp/cospaq_mechanism_047")) / policy_row["policy_id"]
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    policy = normalized_policy(policy_path)
    exporter = VLLM_ROOT / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
    command = [sys.executable, str(exporter), "--model-path", str(MODEL), "--policy-json", str(policy_path), "--output-dir", str(checkpoint), "--cutlass-wrapper-path", str(CUTLASS)]
    if any(method.startswith("sparse_") for method in policy_methods(policy)):
        command.append("--prune")
    evaluator = SOURCE.parent / "scripts/evaluate_runtime_prefill_nll.py"
    log_path = DEBUG / "logs" / f"{policy_row['policy_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            if normalized_policy(checkpoint / "phase_hetero_policy.json") != policy:
                raise RuntimeError("exported policy differs from source policy")
            subprocess.run([sys.executable, str(evaluator), "--checkpoint", str(checkpoint), "--tokenizer", str(MODEL), "--samples", str(SOURCE / "samples/wikitext_2048_targets.pt"), "--output", str(output), "--label", policy_row["policy_id"], "--policy-json", str(policy_path), "--phase-hetero", "--blocks", str(blocks)], check=True, stdout=log, stderr=subprocess.STDOUT)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--selection")
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--one")
    args = parser.parse_args()
    manifest = json.loads((DEBUG / "manifest.json").read_text())
    known = {row["policy_id"]: row for row in manifest}
    if args.one:
        if args.one not in known:
            raise KeyError(args.one)
        one(known[args.one], args.blocks)
        return
    selected = set(args.selection.split(",")) if args.selection else set(known)
    if unknown := selected - set(known):
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    sample_hash = sha256(SOURCE / "samples/wikitext_2048_targets.pt")
    jobs = [row for row in manifest if row["policy_id"] in selected and not complete(DEBUG / "results" / f"{row['policy_id']}.json", row["sha256"], sample_hash, args.blocks)]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    workers: dict[str, tuple[dict, subprocess.Popen]] = {}
    state = {"requested": sorted(selected), "blocks": args.blocks, "completed": [], "failed": []}
    while jobs or workers:
        for gpu in gpus:
            if gpu not in workers and jobs:
                row = jobs.pop(0)
                process = subprocess.Popen([sys.executable, __file__, "--one", row["policy_id"], "--blocks", str(args.blocks)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
                workers[gpu] = (row, process)
        time.sleep(2)
        for gpu, (row, process) in list(workers.items()):
            code = process.poll()
            if code is None:
                continue
            del workers[gpu]
            if code == 0 and complete(DEBUG / "results" / f"{row['policy_id']}.json", row["sha256"], sample_hash, args.blocks):
                state["completed"].append(row["policy_id"])
            else:
                state["failed"].append({"policy": row["policy_id"], "gpu": gpu, "exit_code": code})
            state["pending"] = [row["policy_id"] for row in jobs] + [row[0]["policy_id"] for row in workers.values()]
            (DEBUG / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["failed"]:
        raise RuntimeError(state["failed"])
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
