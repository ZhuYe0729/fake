#!/usr/bin/env python3
"""Driver for linear time proportion study.

Runs bench_qwen3_5_speed.py for each model, testing all configs in one
invocation per model (the bench script handles the sweep internally).

Launch 3 instances in parallel across GPUs 5,6,7.

Usage:
  # Speed phase (no hooks, accurate timing):
  CUDA_VISIBLE_DEVICES=5 python run_study.py --model 2B --phase speed
  CUDA_VISIBLE_DEVICES=6 python run_study.py --model 4B --phase speed
  CUDA_VISIBLE_DEVICES=7 python run_study.py --model 9B --phase speed

  # Breakdown phase (hooks, for time ratio analysis):
  CUDA_VISIBLE_DEVICES=5 python run_study.py --model 2B --phase breakdown
  CUDA_VISIBLE_DEVICES=6 python run_study.py --model 4B --phase breakdown
  CUDA_VISIBLE_DEVICES=7 python run_study.py --model 9B --phase breakdown
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "artifacts/debug/022_linear_time_proportion_study"
BENCH_SCRIPT = REPO_ROOT / "scripts/bench_qwen3_5_speed.py"

MODELS = {
    "2B": "/home/agent/wja/data/models/Qwen/Qwen3.5-2B",
    "4B": "/home/agent/wja/data/models/Qwen/Qwen3.5-4B",
    "9B": "/home/agent/wja/data/models/Qwen/Qwen3.5-9B",
}

# Test matrix: prefill-heavy (short decode) and prefill-decode (long decode)
BATCH_SIZES = [1, 4, 16]
INPUT_TOKENS = [256, 1024, 4096, 8192]
OUTPUT_TOKENS_PREFILL_ONLY = [1]         # prefill-only
OUTPUT_TOKENS_PREFILL_DECODE_SPEED = [32, 128, 256]  # prefill-decode (speed phase)
OUTPUT_TOKENS_PREFILL_DECODE_BREAKDOWN = [32]  # prefill-decode (breakdown phase - reduced)

WARMUP = 3
ITERS = 10


def main():
    parser = argparse.ArgumentParser(description="Linear proportion study driver")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--phase", type=str, required=True, choices=["speed", "breakdown"])
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "prefill_only", "prefill_decode"])
    args = parser.parse_args()

    model_path = MODELS[args.model]
    variant = args.model.lower()
    phase = args.phase

    # Determine output tokens based on scenario and phase
    if args.scenario == "prefill_only":
        output_tokens = OUTPUT_TOKENS_PREFILL_ONLY
    elif args.scenario == "prefill_decode":
        if phase == "breakdown":
            output_tokens = OUTPUT_TOKENS_PREFILL_DECODE_BREAKDOWN
        else:
            output_tokens = OUTPUT_TOKENS_PREFILL_DECODE_SPEED
    else:
        if phase == "breakdown":
            output_tokens = OUTPUT_TOKENS_PREFILL_ONLY + OUTPUT_TOKENS_PREFILL_DECODE_BREAKDOWN
        else:
            output_tokens = OUTPUT_TOKENS_PREFILL_ONLY + OUTPUT_TOKENS_PREFILL_DECODE_SPEED

    # Output paths
    scenario_suffix = args.scenario
    if phase == "speed":
        out_dir = ARTIFACT_DIR / "speed"
        out_csv = out_dir / f"{variant}_speed.csv"
        cmd = [
            sys.executable, str(BENCH_SCRIPT),
            "--model-path", model_path,
            "--variant", args.model,
            "--method", "dense",
            "--dtype", "bf16",
            "--batch-sizes", *[str(x) for x in BATCH_SIZES],
            "--input-tokens", *[str(x) for x in INPUT_TOKENS],
            "--output-tokens", *[str(x) for x in output_tokens],
            "--warmup", str(WARMUP),
            "--iters", str(ITERS),
            "--output-csv", str(out_csv),
        ]
    else:
        out_dir = ARTIFACT_DIR / "breakdown_coarse"
        out_csv = out_dir / f"{variant}_breakdown_coarse_{scenario_suffix}.csv"
        cmd = [
            sys.executable, str(BENCH_SCRIPT),
            "--model-path", model_path,
            "--variant", args.model,
            "--method", "dense",
            "--dtype", "bf16",
            "--batch-sizes", *[str(x) for x in BATCH_SIZES],
            "--input-tokens", *[str(x) for x in INPUT_TOKENS],
            "--output-tokens", *[str(x) for x in output_tokens],
            "--warmup", str(WARMUP),
            "--iters", str(ITERS),
            "--breakdown",
            "--breakdown-mode", "coarse",
            "--output-csv", str(out_csv),
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = ARTIFACT_DIR / "logs" / f"{variant}_{phase}_{args.scenario}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model} ({model_path})")
    print(f"Phase: {phase}")
    print(f"Scenario: {args.scenario}")
    print(f"Output: {out_csv}")
    print(f"Log: {log_file}")
    print(f"Cmd: {' '.join(cmd)}")
    print()

    t0 = time.time()
    with open(log_file, "w") as log_f:
        log_f.write(f"# {args.model} {phase} {args.scenario}\n")
        log_f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"# Cmd: {' '.join(cmd)}\n\n")
        log_f.flush()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()
        proc.wait()
        elapsed = time.time() - t0

        log_f.write(f"\n# Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"# Elapsed: {elapsed:.1f}s\n")
        log_f.write(f"# Return code: {proc.returncode}\n")

    print(f"\nDone. Elapsed: {elapsed:.1f}s, Return code: {proc.returncode}")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()