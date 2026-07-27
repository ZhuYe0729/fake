#!/usr/bin/env python3
"""Restartable multi-GPU dispatcher for the frozen 72-policy NLL sweep."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import CUTLASS, EXPORTER, MODEL, PROTOCOL, RUN, gpu_list, runtime_env, sha256, write_json

EVALUATOR = Path(__file__).with_name("evaluate_decode_nll.py")
VERIFIER = Path(__file__).with_name("verify_checkpoint.py")


def complete(path: Path, policy_hash: str, sample_hash: str, blocks: int) -> bool:
    try:
        result = json.loads(path.read_text())
        runtime = result["runtime"]
        audit = json.loads((RUN / "calibration/audits" / f"{path.stem}.json").read_text())
        waves = (blocks + PROTOCOL["batch"] - 1) // PROTOCOL["batch"]
        return (len(result["blocks"]) == blocks
                and runtime["sample_sha256"] == sample_hash
                and runtime.get("policy_sha256") == policy_hash
                and runtime.get("phase_hetero") is True
                and runtime.get("quantization_config", {}).get("quant_method") == "phase_hetero_mytest"
                and runtime.get("chunked_prefill_enabled") is False
                and runtime.get("max_num_batched_tokens") == PROTOCOL["teacher_forcing_capacity"]
                and runtime.get("phase_trace_events", {}).get("apply_prefill") == waves * 128
                and runtime.get("phase_trace_events", {}).get("apply_decode") == waves * 128 * PROTOCOL["decode_steps"]
                and runtime.get("phase_trace_events", {}).get("enter_decode") == waves
                and runtime.get("phase_trace_events", {}).get("prepare_next_prefill") == max(waves - 1, 0)
                and audit.get("policy_sha256") == policy_hash
                and audit.get("prune") is False)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_one(row: dict, blocks: int) -> None:
    policy = Path(row["path"])
    output = RUN / "calibration/raw" / f"{row['policy_id']}.json"
    samples = RUN / "samples/wikitext_2048_64.pt"
    common = [sys.executable, str(EVALUATOR), "--tokenizer", str(MODEL), "--samples", str(samples),
              "--output", str(output), "--label", row["policy_id"], "--policy-json", str(policy), "--blocks", str(blocks)]
    env = runtime_env()
    temporary = RUN / "temporary/calibration" / row["policy_id"]
    canonical_bf16 = RUN / "canonical/prepared/sparse_bf16/model.pt"
    canonical_nvfp4 = RUN / "canonical/prepared/sparse_nvfp4/model.pt"
    export = [sys.executable, str(EXPORTER), "--model-path", str(MODEL), "--policy-json", str(policy),
              "--output-dir", str(temporary), "--cutlass-wrapper-path", str(CUTLASS),
              "--canonical-sparse-bf16-state", str(canonical_bf16),
              "--canonical-sparse-nvfp4-state", str(canonical_nvfp4)]
    verify = [sys.executable, str(VERIFIER), "--policy", str(policy),
              "--checkpoint", str(temporary), "--canonical-bf16", str(canonical_bf16),
              "--canonical-nvfp4", str(canonical_nvfp4)]
    try:
        if temporary.exists():
            stale_audit = subprocess.run(verify, text=True, capture_output=True, env=env)
            if stale_audit.returncode != 0:
                shutil.rmtree(temporary)
        if not temporary.exists():
            subprocess.run(export, check=True, env=env)
        audit = subprocess.run(verify, check=True, text=True, capture_output=True, env=env)
        audit_path = RUN / "calibration/audits" / f"{row['policy_id']}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(audit.stdout)
        subprocess.run(common[:2] + ["--model", str(temporary)] + common[2:] + [
            "--phase-hetero", "--input-tokens", "2048", "--output-tokens", "64",
            "--batch-size", "8", "--gpu-memory-utilization",
            str(PROTOCOL["gpu_memory_utilization"])], check=True, env=env)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=",".join(gpu_list()))
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--selection")
    parser.add_argument("--one")
    args = parser.parse_args()
    manifest = json.loads((RUN / "policies/prefill_decode/manifest.json").read_text())
    by_id = {row["policy_id"]: row for row in manifest}
    if args.one:
        run_one(by_id[args.one], args.blocks)
        return
    selected = set(args.selection.split(",")) if args.selection else set(by_id)
    unknown = selected - set(by_id)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    sample_hash = sha256(RUN / "samples/wikitext_2048_64.pt")
    jobs = [row for row in manifest if row["policy_id"] in selected and not complete(
        RUN / "calibration/raw" / f"{row['policy_id']}.json", row["sha256"], sample_hash, args.blocks)]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    workers: dict[str, tuple[dict, subprocess.Popen]] = {}
    state = {"requested": sorted(selected), "blocks": args.blocks, "completed": [], "failed": []}
    while jobs or workers:
        for gpu in gpus:
            if gpu not in workers and jobs:
                row = jobs.pop(0)
                env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = gpu
                process = subprocess.Popen([sys.executable, __file__, "--one", row["policy_id"], "--blocks", str(args.blocks)], env=env)
                workers[gpu] = (row, process)
        time.sleep(2)
        for gpu, (row, process) in list(workers.items()):
            if process.poll() is None:
                continue
            del workers[gpu]
            target = RUN / "calibration/raw" / f"{row['policy_id']}.json"
            if process.returncode == 0 and complete(target, row["sha256"], sample_hash, args.blocks):
                state["completed"].append(row["policy_id"])
            else:
                state["failed"].append({"policy": row["policy_id"], "gpu": gpu, "exit_code": process.returncode})
            state["pending"] = [item["policy_id"] for item in jobs] + [item[0]["policy_id"] for item in workers.values()]
            write_json(RUN / "calibration/run_state.json", state)
    if state["failed"]:
        raise RuntimeError(f"failed policies: {state['failed']}")


if __name__ == "__main__":
    main()
