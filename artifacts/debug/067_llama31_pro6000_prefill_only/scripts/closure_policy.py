#!/usr/bin/env python3
"""Export, audit, NLL-score and formally benchmark one policy."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import CUTLASS, EXPORTER, MODEL, PROTOCOL, RUN, runtime_env, write_json

HERE = Path(__file__).resolve().parent


def complete(nll: Path, speed_files: list[Path], blocks: int, runs: int) -> bool:
    try:
        quality = json.loads(nll.read_text())
        events = quality["runtime"]["phase_trace_events"]
        if (len(quality["blocks"]) != blocks or events.get("apply_prefill") != blocks * 128
                or events.get("apply_decode", 0) != 0):
            return False
        rows = [json.loads(path.read_text()) for path in speed_files]
        if any((row["batch"], row["input_seq"], row["output_seq"],
                row["max_num_batched_tokens"], row["chunked_prefill_enabled"],
                row["prefix_caching_enabled"], row["phase_trace_events"].get("apply_decode", 0),
                row["single_process_repeats"])
               != (8, 2048, 1, PROTOCOL["scheduler_capacity"], False, False, 0, True)
               for row in rows):
            return False
        summary = json.loads((speed_files[0].parents[1] / "summary.json").read_text())
        pids = {row["benchmark_process_id"] for row in rows}
        return (len(summary["measured_elapsed_ms"]) == runs
                and summary["single_process_repeats"]
                and summary["warmup_iters"] == 1 and summary["measured_runs"] == runs
                and pids == {summary["benchmark_process_id"]})
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", default=os.environ.get("COSPAQ_SPEED_GPU", "0"))
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=RUN / "closure")
    args = parser.parse_args()
    output = args.output_root / args.label
    raw_speed = output / "speed/raw"
    nll = output / "nll.json"
    required_speed = [raw_speed / "warmup.json"] + [raw_speed / f"measured_{index}.json" for index in range(args.runs)]
    if complete(nll, required_speed, args.blocks, args.runs):
        print(f"already complete: {args.label}")
        return
    checkpoint = RUN / "temporary/closure" / args.label
    canonical_bf16 = RUN / "canonical/prepared/sparse_bf16/model.pt"
    canonical_nvfp4 = RUN / "canonical/prepared/sparse_nvfp4/model.pt"
    env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["COSPAQ_VLLM_NLL_GPU_MEMORY_UTILIZATION"] = os.environ.get("COSPAQ_NLL_GPU_MEMORY_UTILIZATION", "0.70")
    export = [sys.executable, str(EXPORTER), "--model-path", str(MODEL), "--policy-json", str(args.policy),
              "--output-dir", str(checkpoint), "--cutlass-wrapper-path", str(CUTLASS),
              "--canonical-sparse-bf16-state", str(canonical_bf16),
              "--canonical-sparse-nvfp4-state", str(canonical_nvfp4)]
    verify = [sys.executable, str(HERE / "verify_checkpoint.py"), "--policy", str(args.policy),
              "--checkpoint", str(checkpoint), "--canonical-bf16", str(canonical_bf16),
              "--canonical-nvfp4", str(canonical_nvfp4)]
    try:
        if checkpoint.exists():
            stale_audit = subprocess.run(verify, text=True, capture_output=True, env=env)
            if stale_audit.returncode != 0:
                shutil.rmtree(checkpoint)
        if not checkpoint.exists():
            subprocess.run(export, check=True, env=env)
        audit = subprocess.run(verify, check=True, text=True, capture_output=True, env=env)
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoint_audit.json").write_text(audit.stdout)
        if not nll.exists():
            subprocess.run([sys.executable, str(HERE / "evaluate_nll.py"), "--checkpoint", str(checkpoint),
                            "--tokenizer", str(MODEL), "--samples", str(RUN / "samples/wikitext_2048_targets.pt"),
                            "--output", str(nll), "--label", args.label, "--policy-json", str(args.policy),
                            "--phase-hetero", "--blocks", str(args.blocks)], check=True, env=env)
        if raw_speed.exists():
            shutil.rmtree(raw_speed)
        subprocess.run([sys.executable, str(HERE / "benchmark_prefill.py"), "--checkpoint", str(checkpoint),
                        "--batch", str(PROTOCOL["batch"]), "--input-seq", str(PROTOCOL["input_tokens"]),
                        "--output-seq", str(PROTOCOL["output_tokens"]), "--gpu-memory-utilization",
                        str(PROTOCOL["gpu_memory_utilization"]), "--vllm-root", str(__import__("common").VLLM_ROOT),
                        "--cutlass-wrapper-path", str(CUTLASS), "--output-dir", str(raw_speed),
                        "--warmup-iters", "1", "--iters", str(args.runs)], check=True, env=env)
        summary_path = output / "speed/summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update({"label": args.label, "protocol": PROTOCOL})
        write_json(summary_path, summary)
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
