#!/usr/bin/env python3
"""Restartable multi-GPU real-vLLM NLL runner for coverage policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/048_llama2_prefill_quality_coverage"
SOURCE = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat"
MODEL = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(path: Path, policy_hash: str, sample_hash: str, blocks: int) -> bool:
    try:
        payload = json.loads(path.read_text())
        return len(payload["blocks"]) == blocks and payload["runtime"]["policy_sha256"] == policy_hash and payload["runtime"]["sample_sha256"] == sample_hash
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def one(item: dict, blocks: int) -> None:
    output = DEBUG / "results" / f"{item['policy_id']}.json"; policy = Path(item["path"])
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(os.environ.get("COSPAQ_PHASE_TMP_ROOT", "/tmp/cospaq_coverage_048")) / item["policy_id"]
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    content = json.loads(policy.read_text())
    command = [sys.executable, str(VLLM_ROOT / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"), "--model-path", str(MODEL), "--policy-json", str(policy), "--output-dir", str(checkpoint), "--cutlass-wrapper-path", str(CUTLASS)]
    if any(value["prefill_method"].startswith("sparse_") for value in content["method_map"].values()):
        command.append("--prune")
    log = DEBUG / "logs" / f"{item['policy_id']}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
            subprocess.run([sys.executable, str(SOURCE.parent / "scripts/evaluate_runtime_prefill_nll.py"), "--checkpoint", str(checkpoint), "--tokenizer", str(MODEL), "--samples", str(SOURCE / "samples/wikitext_2048_targets.pt"), "--output", str(output), "--label", item["policy_id"], "--policy-json", str(policy), "--phase-hetero", "--blocks", str(blocks)], check=True, stdout=handle, stderr=subprocess.STDOUT)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--gpus", default="1"); parser.add_argument("--blocks", type=int, default=100); parser.add_argument("--one")
    args = parser.parse_args(); manifest = json.loads((DEBUG / "manifest.json").read_text()); known = {item["policy_id"]: item for item in manifest}
    if args.one:
        one(known[args.one], args.blocks); return
    sample_hash = sha256(SOURCE / "samples/wikitext_2048_targets.pt")
    jobs = [item for item in manifest if not complete(DEBUG / "results" / f"{item['policy_id']}.json", item["sha256"], sample_hash, args.blocks)]
    workers: dict[str, tuple[dict, subprocess.Popen]] = {}; gpus = args.gpus.split(","); state = {"blocks": args.blocks, "completed": [], "failed": []}
    while jobs or workers:
        for gpu in gpus:
            if gpu not in workers and jobs:
                item = jobs.pop(0); workers[gpu] = (item, subprocess.Popen([sys.executable, __file__, "--one", item["policy_id"], "--blocks", str(args.blocks)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
        time.sleep(2)
        for gpu, (item, process) in list(workers.items()):
            if process.poll() is None:
                continue
            del workers[gpu]
            (state["completed"] if complete(DEBUG / "results" / f"{item['policy_id']}.json", item["sha256"], sample_hash, args.blocks) else state["failed"]).append(item["policy_id"])
            state["pending"] = [item["policy_id"] for item in jobs] + [item[0]["policy_id"] for item in workers.values()]
            (DEBUG / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["failed"]:
        raise RuntimeError(state["failed"])


if __name__ == "__main__":
    main()
