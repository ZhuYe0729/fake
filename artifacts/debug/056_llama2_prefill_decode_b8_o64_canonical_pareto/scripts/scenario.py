"""Shared immutable protocol for the B=8/S=2048/O=64 experiment."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat"))
MODEL = Path(os.environ.get("COSPAQ_MODEL_PATH", "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"))
CANONICAL = Path(os.environ.get("COSPAQ_CANONICAL_DIR", ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/canonical/prepared"))
CUTLASS = Path(os.environ.get("COSPAQ_CUTLASS_WRAPPER", ROOT / "fake/kernels/cutlass/cutlass_wrapper"))
VLLM = Path(os.environ.get("COSPAQ_VLLM_ROOT", "/home/agent/wja/project/my/cospaq/test/vllm"))

BATCH = 8
INPUT_TOKENS = 2048
OUTPUT_TOKENS = 64
DECODE_STEPS = OUTPUT_TOKENS - 1
GPU_MEMORY_UTILIZATION = 0.80
KV_CACHE_DTYPE = "auto"
MAX_BATCHED_TOKENS = BATCH * INPUT_TOKENS
MAX_MODEL_LEN = INPUT_TOKENS + OUTPUT_TOKENS
