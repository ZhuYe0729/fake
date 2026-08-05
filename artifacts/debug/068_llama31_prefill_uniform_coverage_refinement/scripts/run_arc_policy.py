#!/usr/bin/env python3
"""Evaluate one 068 policy on ARC or the remaining prefill tasks."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess

from scenario import (
    CANONICAL,
    COSPAQ_PYTHON,
    COSPAQ_SPARSE_BF16_EXTENSION_DIR,
    COSPAQ_SPARSE_NVFP4_EXTENSION_DIR,
    EXP,
    MODEL,
    REUSE_DENSE_EXPORTER,
    VLLM_NVFP4_EXTENSION_DIR,
    VLLM_PYTHON,
    VLLM_SPARSE_BF16_EXTENSION_DIR,
    VLLM_SPARSE_NVFP4_EXTENSION_DIR,
)


EVALUATORS = {
    "arc": __file__.replace("run_arc_policy.py", "evaluate_arc_challenge.py"),
    "remaining": __file__.replace("run_arc_policy.py", "evaluate_remaining_prefill.py"),
}
TASKS = {
    "arc": ("arc_challenge",),
    "remaining": ("wikitext", "winogrande", "arc_easy", "mmlu"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--task-set", choices=tuple(TASKS), default="arc")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    policy = EXP / "policies" / f"{args.policy}.json"
    results = [
        EXP / "task_quality/results" / args.policy / task / "full/result.json"
        for task in TASKS[args.task_set]
    ]
    if all(result.exists() for result in results) and not args.force:
        print("\n".join(str(result) for result in results))
        return
    methods = {
        item["prefill_method"]
        for item in json.loads(policy.read_text())["method_map"].values()
    }
    checkpoint = EXP / "temporary/task" / args.policy
    lock = EXP / "temporary/export.lock"
    export_env = os.environ.copy()
    export_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    export_env["CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR"] = str(
        COSPAQ_SPARSE_NVFP4_EXTENSION_DIR
    )
    export_env["CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR"] = str(
        COSPAQ_SPARSE_BF16_EXTENSION_DIR
    )
    task_env = export_env.copy()
    task_env.update({
        "TOKENIZERS_PARALLELISM": "false",
        "COSPAQ_TASK_SAFE_MODE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR": str(VLLM_NVFP4_EXTENSION_DIR),
        "CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR": str(
            VLLM_SPARSE_NVFP4_EXTENSION_DIR
        ),
        "CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR": str(
            VLLM_SPARSE_BF16_EXTENSION_DIR
        ),
    })
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            shutil.rmtree(checkpoint, ignore_errors=True)
            command = [
                str(COSPAQ_PYTHON),
                str(REUSE_DENSE_EXPORTER),
                "--policy-json",
                str(policy),
                "--model-path",
                str(MODEL),
                "--output-dir",
                str(checkpoint),
            ]
            if "sparse_bf16" in methods:
                command += [
                    "--canonical-sparse-bf16-state",
                    str(CANONICAL / "sparse_bf16/model.pt"),
                ]
            if "sparse_nvfp4" in methods:
                command += [
                    "--canonical-sparse-nvfp4-state",
                    str(CANONICAL / "sparse_nvfp4/model.pt"),
                ]
            command.append("--force")
            subprocess.run(command, check=True, env=export_env)

        manifest = {
            "model_path": str(MODEL),
            "policies": [{
                "label": args.policy,
                "kind": "ours",
                "checkpoint": str(checkpoint),
                "policy_json": str(policy),
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            }],
        }
        manifest_path = EXP / "task_quality/manifests" / f"{args.policy}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run(
            [
                str(VLLM_PYTHON),
                EVALUATORS[args.task_set],
                "--manifest",
                str(manifest_path),
                "--policy",
                args.policy,
                "--output-root",
                str(EXP / "task_quality/results"),
            ],
            check=True,
            env=task_env,
        )
    finally:
        shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
