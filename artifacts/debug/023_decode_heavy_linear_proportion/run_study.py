#!/usr/bin/env python3
"""Driver for decode-heavy linear proportion study.

Tests short-prefill + long-decode scenarios to measure linear proportion
in decode-dominated workloads.

Test matrix:
  Models: 2B, 4B, 9B
  Batch sizes: 1, 4, 16
  Input tokens: 4, 16, 64, 256 (short prefill)
  Output tokens: 128, 256, 512 (long decode)

Usage:
  CUDA_VISIBLE_DEVICES=5 python run_study.py --model 2B --phase speed
  CUDA_VISIBLE_DEVICES=6 python run_study.py --model 4B --phase speed
  CUDA_VISIBLE_DEVICES=7 python run_study.py --model 9B --phase speed

  CUDA_VISIBLE_DEVICES=5 python run_study.py --model 2B --phase breakdown
  ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "artifacts/debug/023_decode_heavy_linear_proportion"
BENCH_SCRIPT = REPO_ROOT / "scripts/bench_qwen3_5_speed.py"

MODELS = {
    "2B": "/home/agent/wja/data/models/Qwen/Qwen3.5-2B",
    "4B": "/home/agent/wja/data/models/Qwen/Qwen3.5-4B",
    "9B": "/home/agent/wja/data/models/Qwen/Qwen3.5-9B",
}

BATCH_SIZES = [1, 4, 16]
INPUT_TOKENS = [4, 16, 64, 256]          # short prefill
OUTPUT_TOKENS_SPEED = [128, 256, 512]     # long decode
OUTPUT_TOKENS_BREAKDOWN = [128]           # representative decode length

WARMUP = 3
ITERS = 10


def main():
    parser = argparse.ArgumentParser(description="Decode-heavy linear proportion study")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--phase", type=str, required=True, choices=["speed", "breakdown"])
    args = parser.parse_args()

    model_path = MODELS[args.model]
    variant = args.model.lower()
    phase = args.phase

    out_dir = ARTIFACT_DIR / phase if phase == "speed" else ARTIFACT_DIR / "breakdown_coarse"
    out_dir.mkdir(parents=True, exist_ok=True)

    if phase == "speed":
        out_csv = out_dir / f"{variant}_speed.csv"
        output_tokens = OUTPUT_TOKENS_SPEED
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
        out_csv = out_dir / f"{variant}_breakdown_coarse.csv"
        output_tokens = OUTPUT_TOKENS_BREAKDOWN
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

    log_file = ARTIFACT_DIR / "logs" / f"{variant}_{phase}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    n_configs = len(BATCH_SIZES) * len(INPUT_TOKENS) * len(output_tokens)
    print(f"Model: {args.model} ({model_path})")
    print(f"Phase: {phase}")
    print(f"Configs: {n_configs} ({len(BATCH_SIZES)} batches * {len(INPUT_TOKENS)} inputs * {len(output_tokens)} outputs)")
    print(f"Output: {out_csv}")
    print(f"Log: {log_file}")
    print()

    t0 = time.time()
    with open(log_file, "w") as log_f:
        log_f.write(f"# {args.model} {phase} (decode-heavy)\n")
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


if __name__ == "__main__":
    main()