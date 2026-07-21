#!/usr/bin/env python3
"""Run resumable canonical teacher-forced NLL labels across available GPUs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat"
MODEL = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
STREAM = ROOT / "artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/stream_phase_policy_nll.py"
CANONICAL = ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/canonical/prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--policy-ids", help="Comma-separated subset; default is all 72.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((EXP / "policies/prefill_decode/manifest.json").read_text())
    wanted = set(args.policy_ids.split(",")) if args.policy_ids else None
    rows = [row for row in manifest if wanted is None or row["policy_id"] in wanted]
    out = EXP / "nll/raw"; logs = ROOT / "artifacts/debug/055_llama2_prefill_decode_canonical_pareto/logs/nll_full"
    out.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    todo = [row for row in rows if not (out / f"{row['policy_id']}.json").exists()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]

    temporary = out / "temporary_checkpoints"

    def run(gpu: str, row: dict[str, str]) -> str:
        policy_id = row["policy_id"]
        # A failed prior attempt can leave disposable checkpoint/capture
        # directories. A result JSON is the completion marker, so cleaning
        # these per-policy intermediates is safe here.
        shutil.rmtree(temporary / policy_id, ignore_errors=True)
        shutil.rmtree(out / f"{policy_id}_captures", ignore_errors=True)
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
        command = [sys.executable, str(STREAM), "--model-path", str(MODEL), "--tokenizer", str(MODEL),
                   "--policy-json", row["path"], "--samples", str(EXP / "samples/wikitext_2048_80.pt"),
                   "--output", str(out / f"{policy_id}.json"), "--label", policy_id,
                   "--blocks", str(args.blocks), "--canonical-sparse-bf16-state",
                   str(CANONICAL / "sparse_bf16/model.pt"), "--canonical-sparse-nvfp4-state",
                   str(CANONICAL / "sparse_nvfp4/model.pt")]
        with (logs / f"{policy_id}.log").open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)
        return policy_id

    def worker(gpu: str, assigned: list[dict[str, str]]) -> list[str]:
        return [run(gpu, row) for row in assigned]

    assignments = [todo[index::len(gpus)] for index in range(len(gpus))]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, rows) for gpu, rows in zip(gpus, assignments) if rows]
        for future in as_completed(futures):
            try:
                for policy_id in future.result():
                    print(policy_id, flush=True)
            except subprocess.CalledProcessError as error:
                failures.append(str(error.cmd[error.cmd.index("--label") + 1]))
    if failures:
        raise RuntimeError(f"failed policies: {failures}")
    print(json.dumps({"completed": len(rows), "new": len(todo), "blocks": args.blocks}))


if __name__ == "__main__":
    main()
