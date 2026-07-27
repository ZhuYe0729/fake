#!/usr/bin/env python3
"""Single restartable entry point for every 065 experiment stage."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from common import COSPAQ_PYTHON, RUN, VLLM_PYTHON, gpu_list, runtime_env

HERE = Path(__file__).resolve().parent


def run(python: Path, script: str, *args: str, env=None) -> None:
    subprocess.run([str(python), str(HERE / script), *args], check=True, env=env or runtime_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "bootstrap", "canonical", "prewarm", "local-errors", "smoke",
                                          "nll", "fit", "profile", "solve", "closure", "task-data",
                                          "select-tasks", "tasks", "consolidate", "validate"))
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--use-local-proxy", action="store_true")
    parser.add_argument("--gpus", default=",".join(gpu_list()))
    parser.add_argument("--speed-gpu", default=os.environ.get("COSPAQ_SPEED_GPU", "0"))
    args = parser.parse_args()
    if args.stage == "preflight":
        run(COSPAQ_PYTHON, "preflight.py", *(["--no-gpu"] if args.no_gpu else []))
    elif args.stage == "bootstrap":
        run(COSPAQ_PYTHON, "bootstrap.py", "--resume")
        run(COSPAQ_PYTHON, "validate_decode.py", "bootstrap")
        run(COSPAQ_PYTHON, "validate_decode.py", "isolation")
    elif args.stage == "canonical":
        run(COSPAQ_PYTHON, "copy_canonical.py", "--resume")
        run(COSPAQ_PYTHON, "verify_canonical_sparse.py")
        run(COSPAQ_PYTHON, "validate_decode.py", "canonical")
    elif args.stage == "prewarm":
        env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = args.speed_gpu
        run(VLLM_PYTHON, "prewarm_phase_extensions.py", env=env)
    elif args.stage == "local-errors":
        run(COSPAQ_PYTHON, "run_local_errors.py", "--gpus", args.gpus, "--blocks", "16")
        run(COSPAQ_PYTHON, "validate_decode.py", "local-errors")
    elif args.stage == "nll":
        run(VLLM_PYTHON, "run_calibration_nll.py", "--gpus", args.gpus, "--blocks", "100")
        run(COSPAQ_PYTHON, "merge_nll.py")
        run(COSPAQ_PYTHON, "validate_decode.py", "nll")
    elif args.stage == "smoke":
        run(VLLM_PYTHON, "run_smoke.py", "--gpus", args.gpus)
    elif args.stage == "fit":
        run(COSPAQ_PYTHON, "build_coverage_holdout.py")
        run(COSPAQ_PYTHON, "fit_phase_quality.py", "--split-json",
            str(RUN / "policies/prefill_decode/coverage_holdout.json"),
            "--report-name", "quality")
    elif args.stage == "profile":
        run(COSPAQ_PYTHON, "profile_kernels.py", "--gpu", args.speed_gpu)
    elif args.stage == "solve":
        run(COSPAQ_PYTHON, "audit_speed_actions.py", "--predictor-root",
            str(RUN / "kernel_profile/modeling"))
        run(COSPAQ_PYTHON, "solve_phase_pareto.py")
        run(COSPAQ_PYTHON, "validate_decode.py", "profile")
    elif args.stage == "closure":
        run(VLLM_PYTHON, "run_closure.py", "--gpu", args.speed_gpu, "--runs", "5", "--blocks", "100")
        run(COSPAQ_PYTHON, "validate_decode.py", "closure")
    elif args.stage == "task-data":
        run(VLLM_PYTHON, "prepare_pmpd_data.py")
    elif args.stage == "select-tasks":
        run(COSPAQ_PYTHON, "select_task_policies.py")
    elif args.stage == "tasks":
        run(VLLM_PYTHON, "run_pmpd_tasks.py", "--gpus", args.gpus)
        run(VLLM_PYTHON, "merge_pmpd_tasks.py")
        run(COSPAQ_PYTHON, "validate_decode.py", "tasks")
    elif args.stage == "consolidate":
        run(COSPAQ_PYTHON, "consolidate_decode.py")
        run(COSPAQ_PYTHON, "validate_decode.py", "results")
    elif args.stage == "validate":
        run(COSPAQ_PYTHON, "validate_decode.py", "all")


if __name__ == "__main__":
    main()
