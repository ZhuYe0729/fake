"""Shared constants and paths for the real-vLLM prefill calibration bundle."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = Path(os.environ.get("COSPAQ_PREFILL_EXPERIMENT_ROOT", ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration"))
VLLM_ROOT = Path(os.environ.get("COSPAQ_VLLM_ROOT", "/home/agent/wja/project/my/cospaq/test/vllm"))
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")

MODELS = {
    "llama2": {
        "directory": "llama2_7b_chat",
        "path": Path(os.environ.get("COSPAQ_MODEL_PATH", "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")),
        "uniform_root": ROOT / "artifacts/exports/vllm/baselines/llama2-7b-chat/checkpoints",
        "local_error_source": ROOT / "artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv",
    },
    "llama31": {
        "directory": "llama31_8b_instruct",
        "path": Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"),
        "uniform_root": ROOT / "artifacts/exports/vllm/baselines/llama3.1-8b-instruct/checkpoints",
        "local_error_source": ROOT / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto/local_errors",
    },
}


def model_root(model: str) -> Path:
    return DEBUG / MODELS[model]["directory"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    policy.setdefault("modules_to_not_convert", ["lm_head"])
    return policy
