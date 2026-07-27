#!/usr/bin/env python3
"""Single restartable entry point for every 064 experiment stage."""
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
        run(COSPAQ_PYTHON, "validate.py", "bootstrap")
        run(COSPAQ_PYTHON, "validate.py", "isolation")
    elif args.stage == "canonical":
        available = [value for value in args.gpus.split(",") if value]
        commands = (
            [str(COSPAQ_PYTHON), str(HERE / "prepare_canonical_sparse.py"), "--methods", "sparse_bf16",
             "--gpu", available[0], "--skip-existing"],
            [str(COSPAQ_PYTHON), str(HERE / "prepare_canonical_sparse.py"), "--methods", "sparse_nvfp4",
             "--gpu", available[min(1, len(available) - 1)], "--sparse-nvfp4-prequant-only", "--skip-existing"],
        )
        processes = [subprocess.Popen(command, env=runtime_env()) for command in commands]
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"canonical workers failed: {codes}")
        run(COSPAQ_PYTHON, "verify_canonical_sparse.py")
        run(COSPAQ_PYTHON, "validate.py", "canonical")
    elif args.stage == "prewarm":
        env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = args.speed_gpu
        run(VLLM_PYTHON, "prewarm_phase_extensions.py", env=env)
    elif args.stage == "local-errors":
        available = [value for value in args.gpus.split(",") if value]
        methods = ("sparse_bf16", "sparse_nvfp4")
        processes = []
        for index, method in enumerate(methods):
            gpu = available[index % len(available)]
            processes.append(subprocess.Popen([str(COSPAQ_PYTHON), str(HERE / "collect_local_errors.py"),
                                               "--method", method, "--gpu", gpu, "--blocks", "16"], env=runtime_env()))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"local-error workers failed: {codes}")
        run(COSPAQ_PYTHON, "assemble_local_errors.py")
        run(COSPAQ_PYTHON, "validate.py", "local-errors")
    elif args.stage == "nll":
        run(VLLM_PYTHON, "run_calibration_nll.py", "--gpus", args.gpus, "--blocks", "100")
        run(COSPAQ_PYTHON, "merge_nll.py")
        run(COSPAQ_PYTHON, "validate.py", "nll")
    elif args.stage == "smoke":
        run(VLLM_PYTHON, "run_smoke.py", "--gpus", args.gpus)
    elif args.stage == "fit":
        run(COSPAQ_PYTHON, "fit_quality.py")
    elif args.stage == "profile":
        run(COSPAQ_PYTHON, "profile_kernels.py", "--gpu", args.speed_gpu)
    elif args.stage == "solve":
        run(COSPAQ_PYTHON, "solve_pareto.py", "--predictor-root", str(RUN / "kernel_profile/modeling"))
        run(COSPAQ_PYTHON, "validate.py", "profile")
    elif args.stage == "closure":
        run(VLLM_PYTHON, "run_closure.py", "--gpu", args.speed_gpu, "--runs", "5", "--blocks", "100")
        run(COSPAQ_PYTHON, "validate.py", "closure")
    elif args.stage == "task-data":
        proxy = ["--use-local-proxy"] if args.use_local_proxy else []
        run(VLLM_PYTHON, "prepare_task_data.py", *proxy)
    elif args.stage == "select-tasks":
        run(COSPAQ_PYTHON, "select_task_policies.py")
    elif args.stage == "tasks":
        run(VLLM_PYTHON, "run_tasks.py", "--gpus", args.gpus)
        run(COSPAQ_PYTHON, "validate.py", "tasks")
    elif args.stage == "consolidate":
        run(COSPAQ_PYTHON, "consolidate.py")
        run(COSPAQ_PYTHON, "validate.py", "results")
    elif args.stage == "validate":
        run(COSPAQ_PYTHON, "validate.py", "all")


if __name__ == "__main__":
    main()
