#!/usr/bin/env python3
"""Run one phase policy through the final fresh-process speed protocol."""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
from pathlib import Path

from config import (
    CANONICAL,
    DEBUG,
    DECODE,
    DECODE_EXPORTER,
    DECODE_SPEED_RUNNER,
    DECODE_VERIFIER,
    MARLIN_EXTENSION,
    MODEL,
    RUNTIME_TMP,
    VLLM_PYTHON,
    candidate_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--policy-json")
    parser.add_argument("--output-name")
    args = parser.parse_args()
    policy = candidate_path(DECODE, args.policy) if args.policy_json is None else Path(args.policy_json)
    output_name = args.output_name or args.policy
    summary = DECODE / f"speed/runs/{output_name}/summary.csv"
    if summary.exists():
        print(summary)
        return
    RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
    stage_lock = DEBUG / "temporary/speed_stage.lock"
    stage_lock.parent.mkdir(parents=True, exist_ok=True)
    stage_handle = stage_lock.open("w")
    fcntl.flock(stage_handle, fcntl.LOCK_EX)
    run_lock = DECODE / f"temporary/locks/speed_{output_name}.lock"
    run_lock.parent.mkdir(parents=True, exist_ok=True)
    run_handle = run_lock.open("w")
    fcntl.flock(run_handle, fcntl.LOCK_EX)
    if summary.exists():
        print(summary)
        run_handle.close()
        stage_handle.close()
        return
    env = os.environ.copy()
    env.update({
        "COSPAQ_EXPERIMENT_ROOT": str(DEBUG),
        "COSPAQ_EXPERIMENT_DIR": str(DECODE),
        "COSPAQ_MODEL_PATH": str(MODEL),
        "COSPAQ_CANONICAL_DIR": str(CANONICAL),
        "COSPAQ_EXPORTER": str(DECODE_EXPORTER),
        "COSPAQ_SPEED_RUNNER": str(DECODE_SPEED_RUNNER),
        "COSPAQ_VERIFY_CHECKPOINT": str(DECODE_VERIFIER),
        "VLLM_PYTHON": str(VLLM_PYTHON),
        "TMPDIR": str(RUNTIME_TMP),
        "CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR": str(MARLIN_EXTENSION),
    })
    try:
        subprocess.run([
            "bash", str(DECODE_SPEED_RUNNER), output_name, str(policy), args.gpu,
        ], check=True, env=env)
    finally:
        shutil.rmtree(DECODE / f"speed/runs/{output_name}/checkpoint", ignore_errors=True)
        run_handle.close()
        stage_handle.close()


if __name__ == "__main__":
    main()
