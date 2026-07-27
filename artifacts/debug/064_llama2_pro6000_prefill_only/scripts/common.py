#!/usr/bin/env python3
"""Shared paths, protocol constants, hashes and subprocess helpers for 064."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

BUNDLE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("COSPAQ_REPO_ROOT", BUNDLE.parents[2])).resolve()
RUN = BUNDLE / "runs/experiment"
INPUTS = BUNDLE / "inputs"
RESULTS = BUNDLE / "results"
VALIDATION = BUNDLE / "validation"
MODEL = Path(os.environ.get("COSPAQ_MODEL_PATH", "/root/data/models/Llama-2-7b-chat-hf")).resolve()
VLLM_ROOT = Path(os.environ.get("COSPAQ_VLLM_ROOT", "/root/workspaces/cospaq/vllm-cospaq")).resolve()
CUTLASS = REPO / "fake/kernels/cutlass/cutlass_wrapper"
EXPORTER = BUNDLE / "scripts/export_phase_hetero_model.py"
COSPAQ_PYTHON = Path(os.environ.get("COSPAQ_COSPAQ_PYTHON", "/root/miniconda3/envs/cospaq/bin/python"))
VLLM_PYTHON = Path(os.environ.get("COSPAQ_VLLM_PYTHON", "/root/miniconda3/envs/vllm/bin/python"))
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
PROTOCOL = {
    "batch": 8,
    "input_tokens": 2048,
    "output_tokens": 1,
    "vllm_v1": True,
    "kv_cache_dtype": "auto",
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
    "gpu_memory_utilization": float(os.environ.get("COSPAQ_GPU_MEMORY_UTILIZATION", "0.90")),
    "warmups": 1,
    "measured_runs": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def read_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    policy.setdefault("modules_to_not_convert", ["lm_head"])
    return policy


normalized_policy = read_policy


def gpu_list() -> list[str]:
    return [value.strip() for value in os.environ.get("COSPAQ_GPUS", "0").split(",") if value.strip()]


def command_output(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = [str(VLLM_ROOT / "vllm"), str(VLLM_ROOT), str(CUTLASS), str(REPO)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    env["NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    env["MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    env["SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    env["SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH"] = str(CUTLASS)
    task_cache = Path(env.get("COSPAQ_TASK_CACHE", BUNDLE / "cache/huggingface"))
    if env.get("COSPAQ_TASK_CACHE"):
        env["HF_HOME"] = str(task_cache)
        env["HF_DATASETS_CACHE"] = str(task_cache / "datasets")
    env.setdefault("MPLCONFIGDIR", str(BUNDLE / "cache/matplotlib"))
    return env
