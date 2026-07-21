"""Immutable paths and protocol for Llama3 canonical prefill-only closure."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/llama31_8b_instruct"
MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
VLLM = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
SOURCE_038 = ROOT / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto"
SOURCE_057 = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
CANONICAL = SOURCE_057 / "canonical/prepared"
LOCAL_ERRORS = SOURCE_057 / "local_errors"
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
KERNEL = {"dense_bf16": "dense_bf16", "dense_nvfp4": "dense_nvfp4", "sparse_bf16": "sparse_bf16", "sparse_nvfp4": "sparse_nvfp4", "w4a16_ours": "marlin_nvfp4"}
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
BATCH, INPUT_TOKENS, REPEATS = 8, 2048, 5
