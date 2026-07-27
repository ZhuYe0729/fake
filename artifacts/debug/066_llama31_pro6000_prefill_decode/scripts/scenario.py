"""Compatibility names for the artifact-local prefill-decode modules."""
from common import BUNDLE, CUTLASS, MODEL, PROTOCOL, RUN, VLLM_ROOT

ROOT = BUNDLE.parents[2]
EXP = RUN
CANONICAL = RUN / "canonical/prepared"
VLLM = VLLM_ROOT
BATCH = PROTOCOL["batch"]
INPUT_TOKENS = PROTOCOL["input_tokens"]
OUTPUT_TOKENS = PROTOCOL["output_tokens"]
DECODE_STEPS = PROTOCOL["decode_steps"]
GPU_MEMORY_UTILIZATION = PROTOCOL["gpu_memory_utilization"]
KV_CACHE_DTYPE = PROTOCOL["kv_cache_dtype"]
MAX_BATCHED_TOKENS = PROTOCOL["teacher_forcing_capacity"]
MAX_MODEL_LEN = INPUT_TOKENS + OUTPUT_TOKENS
