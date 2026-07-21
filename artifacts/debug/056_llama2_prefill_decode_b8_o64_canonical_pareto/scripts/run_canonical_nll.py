#!/usr/bin/env python3
"""Run resumable canonical teacher-forced NLL labels across available GPUs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
from scenario import EXP, MODEL, CANONICAL, INPUT_TOKENS, OUTPUT_TOKENS
STREAM = ROOT / "artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/stream_phase_policy_nll.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--startup-stagger-seconds", type=float, default=12.0)
    parser.add_argument("--policy-ids", help="Comma-separated subset; default is all 72.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((EXP / "policies/prefill_decode/manifest.json").read_text())
    wanted = set(args.policy_ids.split(",")) if args.policy_ids else None
    rows = [row for row in manifest if wanted is None or row["policy_id"] in wanted]
    out = EXP / "nll/raw"; logs = EXP.parent.parent / "logs/nll_full"
    out.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    def complete(row: dict[str, str]) -> bool:
        path = out / f"{row['policy_id']}.json"
        if not path.exists():
            return False
        try:
            runtime = json.loads(path.read_text())["runtime"]
            return (runtime.get("input_tokens") == INPUT_TOKENS and
                    runtime.get("decode_tokens") == OUTPUT_TOKENS and
                    runtime.get("batch_size") == 8 and
                    runtime.get("blocks") == args.blocks and
                    runtime.get("max_num_batched_tokens") == 8 * (INPUT_TOKENS + OUTPUT_TOKENS))
        except (KeyError, json.JSONDecodeError):
            return False
    todo = [row for row in rows if not complete(row)]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]

    temporary = out / "temporary_checkpoints"

    def run(gpu: str, row: dict[str, str]) -> str:
        policy_id = row["policy_id"]
        # A failed prior attempt can leave disposable checkpoint/capture
        # directories. A result JSON is the completion marker, so cleaning
        # these per-policy intermediates is safe here.
        shutil.rmtree(temporary / policy_id, ignore_errors=True)
        shutil.rmtree(out / f"{policy_id}_captures", ignore_errors=True)
        for stale in (out / f"{policy_id}.json", out / f"{policy_id}.export_provenance.json",
                      out / f"{policy_id}.phase_trace.json"):
            stale.unlink(missing_ok=True)
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
        command = [sys.executable, str(STREAM), "--model-path", str(MODEL), "--tokenizer", str(MODEL),
                   "--policy-json", row["path"], "--samples", str(EXP / "samples/wikitext_2048_64.pt"),
                   "--output", str(out / f"{policy_id}.json"), "--label", policy_id,
                   "--blocks", str(args.blocks), "--canonical-sparse-bf16-state",
                   str(CANONICAL / "sparse_bf16/model.pt"), "--canonical-sparse-nvfp4-state",
                   str(CANONICAL / "sparse_nvfp4/model.pt"), "--input-tokens", str(INPUT_TOKENS),
                   "--output-tokens", str(OUTPUT_TOKENS), "--batch-size", "8",
                   "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        with (logs / f"{policy_id}.log").open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)
        return policy_id

    def worker(gpu: str, assigned: list[dict[str, str]], startup_delay: float) -> list[str]:
        if startup_delay:
            time.sleep(startup_delay)
        return [run(gpu, row) for row in assigned]

    assignments = [todo[index::len(gpus)] for index in range(len(gpus))]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, rows, index * args.startup_stagger_seconds)
                   for index, (gpu, rows) in enumerate(zip(gpus, assignments)) if rows]
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
