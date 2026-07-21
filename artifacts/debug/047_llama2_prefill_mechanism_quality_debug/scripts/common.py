"""Shared paths and policy helpers for the mechanism-quality debug bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/047_llama2_prefill_mechanism_quality_debug"
SOURCE = ROOT / "artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat"
MODEL = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
ERRORS = ROOT / "artifacts/debug/007_llama2_quality_modeling/sensitivity/module_method_errors.csv"
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4")
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
PARTS = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "o_proj": ("o_proj",), "gate_up_proj": ("gate_proj", "up_proj"), "down_proj": ("down_proj",)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_methods(policy: dict) -> list[str]:
    values = [policy["default_prefill_method"], policy["default_decode_method"]]
    values.extend(value for entry in policy["method_map"].values() for value in entry.values())
    return values


def normalized_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    policy.setdefault("modules_to_not_convert", ["lm_head"])
    return policy
