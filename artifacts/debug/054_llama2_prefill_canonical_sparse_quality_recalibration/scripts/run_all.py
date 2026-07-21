#!/usr/bin/env python3
"""Restartable real-vLLM NLL dispatcher using canonical calibrated sparse states."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"))
MODEL = Path(os.environ.get("COSPAQ_MODEL_PATH", "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"))
VLLM_ROOT = Path(os.environ.get("COSPAQ_VLLM_ROOT", "/home/agent/wja/project/my/cospaq/test/vllm"))
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
EVALUATOR = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_runtime_prefill_nll.py"
EXPORTER = VLLM_ROOT / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
CANONICAL_BF16 = EXPERIMENT / "canonical/prepared/sparse_bf16/model.pt"
CANONICAL_NVFP4 = EXPERIMENT / "canonical/prepared/sparse_nvfp4/model.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(path: Path, policy_hash: str, sample_hash: str, blocks: int) -> bool:
    try:
        result = json.loads(path.read_text())
        return (len(result["blocks"]) == blocks and result["runtime"]["sample_sha256"] == sample_hash
                and result["runtime"].get("policy_sha256") == policy_hash)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def one(row: dict, blocks: int) -> None:
    policy = Path(row["path"])
    output = EXPERIMENT / "results" / f"{row['policy_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    samples = EXPERIMENT / "samples/wikitext_2048_targets.pt"
    common = [sys.executable, str(EVALUATOR), "--tokenizer", str(MODEL), "--samples", str(samples),
              "--output", str(output), "--label", row["policy_id"], "--policy-json", str(policy),
              "--blocks", str(blocks)]
    if row["policy_id"] == "p00":
        subprocess.run(common[:2] + ["--checkpoint", str(MODEL)] + common[2:], check=True)
        return
    if not CANONICAL_BF16.exists() or not CANONICAL_NVFP4.exists():
        raise FileNotFoundError("canonical sparse states are required before NLL evaluation")
    temporary = Path(os.environ.get("COSPAQ_PHASE_TMP_ROOT", "/tmp/cospaq_054_phase")) / row["policy_id"]
    if temporary.exists():
        raise FileExistsError(temporary)
    export = [sys.executable, str(EXPORTER), "--model-path", str(MODEL), "--policy-json", str(policy),
              "--output-dir", str(temporary), "--cutlass-wrapper-path", str(CUTLASS),
              "--canonical-sparse-bf16-state", str(CANONICAL_BF16),
              "--canonical-sparse-nvfp4-state", str(CANONICAL_NVFP4)]
    try:
        subprocess.run(export, check=True)
        subprocess.run(common[:2] + ["--checkpoint", str(temporary)] + common[2:] + ["--phase-hetero"], check=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--selection")
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--one")
    args = parser.parse_args()
    manifest = json.loads((EXPERIMENT / "policies/prefill_only/manifest.json").read_text())
    by_id = {row["policy_id"]: row for row in manifest}
    if args.one:
        one(by_id[args.one], args.blocks)
        return
    selected = set(args.selection.split(",")) if args.selection else set(by_id)
    unknown = selected - set(by_id)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    sample_hash = sha256(EXPERIMENT / "samples/wikitext_2048_targets.pt")
    jobs = [row for row in manifest if row["policy_id"] in selected and not complete(
        EXPERIMENT / "results" / f"{row['policy_id']}.json", row["sha256"], sample_hash, args.blocks)]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    workers: dict[str, tuple[dict, subprocess.Popen]] = {}
    state = {"requested": sorted(selected), "blocks": args.blocks, "pending": [row["policy_id"] for row in jobs], "completed": [], "failed": []}
    while jobs or workers:
        for gpu in gpus:
            if gpu not in workers and jobs:
                row = jobs.pop(0)
                process = subprocess.Popen([sys.executable, __file__, "--one", row["policy_id"], "--blocks", str(args.blocks)], env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
                workers[gpu] = (row, process)
        time.sleep(2)
        for gpu, (row, process) in list(workers.items()):
            if process.poll() is None:
                continue
            del workers[gpu]
            result = EXPERIMENT / "results" / f"{row['policy_id']}.json"
            if process.returncode == 0 and complete(result, row["sha256"], sample_hash, args.blocks):
                state["completed"].append(row["policy_id"])
            else:
                state["failed"].append({"policy": row["policy_id"], "gpu": gpu, "exit_code": process.returncode})
            state["pending"] = [item["policy_id"] for item in jobs] + [item[0]["policy_id"] for item in workers.values()]
            (EXPERIMENT / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["failed"]:
        raise RuntimeError(f"failed policies: {state['failed']}")
    print(json.dumps({"completed": len(state["completed"]), "skipped": len(selected) - len(state["completed"])}, indent=2))


if __name__ == "__main__":
    main()
