#!/usr/bin/env python3
"""Run resumable real phase-vLLM NLL labels for the 18 extension policies."""
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
OUT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
STREAM = ROOT / "artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/stream_phase_policy_nll.py"
CANONICAL = SOURCE / "canonical/prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--startup-stagger-seconds", type=float, default=12.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((OUT / "extension/policies/manifest.json").read_text())
    raw = OUT / "extension/nll/raw"
    logs = OUT / "extension/logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    def complete(row: dict[str, object]) -> bool:
        path = raw / f"{row['policy_id']}.json"
        if not path.exists():
            return False
        try:
            runtime = json.loads(path.read_text())["runtime"]
            return (
                runtime.get("input_tokens") == 2048
                and runtime.get("decode_tokens") == 64
                and runtime.get("batch_size") == 8
                and runtime.get("blocks") == args.blocks
                and runtime.get("max_num_batched_tokens") == 16896
            )
        except (KeyError, json.JSONDecodeError):
            return False

    todo = [row for row in manifest if not complete(row)]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    temporary = raw / "temporary_checkpoints"

    def run(gpu: str, row: dict[str, object]) -> str:
        policy_id = str(row["policy_id"])
        shutil.rmtree(temporary / policy_id, ignore_errors=True)
        shutil.rmtree(raw / f"{policy_id}_captures", ignore_errors=True)
        for suffix in (".json", ".export_provenance.json", ".phase_trace.json"):
            (raw / f"{policy_id}{suffix}").unlink(missing_ok=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [
            sys.executable, str(STREAM), "--model-path", str(MODEL), "--tokenizer", str(MODEL),
            "--policy-json", str(row["path"]), "--samples", str(SOURCE / "samples/wikitext_2048_64.pt"),
            "--output", str(raw / f"{policy_id}.json"), "--label", policy_id,
            "--blocks", str(args.blocks), "--canonical-sparse-bf16-state",
            str(CANONICAL / "sparse_bf16/model.pt"), "--canonical-sparse-nvfp4-state",
            str(CANONICAL / "sparse_nvfp4/model.pt"), "--input-tokens", "2048",
            "--output-tokens", "64", "--batch-size", "8", "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        with (logs / f"{policy_id}.log").open("w") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        return policy_id

    def worker(gpu: str, assigned: list[dict[str, object]], delay: float) -> list[str]:
        if delay:
            time.sleep(delay)
        return [run(gpu, row) for row in assigned]

    assignments = [todo[index::len(gpus)] for index in range(len(gpus))]
    failures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, assigned, index * args.startup_stagger_seconds)
            for index, (gpu, assigned) in enumerate(zip(gpus, assignments)) if assigned
        ]
        for future in as_completed(futures):
            try:
                for policy_id in future.result():
                    print(policy_id, flush=True)
            except subprocess.CalledProcessError as error:
                failures.append(str(error.cmd[error.cmd.index("--label") + 1]))
    if failures:
        raise RuntimeError(f"failed policies: {failures}")
    print(json.dumps({"policies": len(manifest), "new": len(todo), "blocks": args.blocks}))


if __name__ == "__main__":
    main()
