"""Frozen inputs and protocol for debug 068."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/068_llama31_prefill_uniform_coverage_refinement"
EXP = DEBUG / "llama31_8b_instruct"
SOURCE_058 = ROOT / "artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/llama31_8b_instruct"
SOURCE_038 = ROOT / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto"
SOURCE_057 = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
SOURCE_060 = ROOT / "artifacts/debug/060_two_model_two_scenario_result_consolidation/llama31_8b_instruct/prefill_only_02"
SOURCE_061 = ROOT / "artifacts/debug/061_llama31_prefill_warmed_speed_revalidation"
SOURCE_061_MODEL = SOURCE_061 / "llama31_8b_instruct"
MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")

COSPAQ_PYTHON = Path("/home/agent/wja/miniconda3/envs/cospaq/bin/python")
VLLM_PYTHON = Path("/home/agent/wja/miniconda3/envs/vllm/bin/python")
REUSE_DENSE_EXPORTER = SOURCE_061 / "scripts/export_phase_reuse_dense_nvfp4.py"
CANONICAL = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/canonical/prepared"
BENCH = ROOT / "artifacts/debug/059_llama31_prefill_speed_decomposition/scripts/benchmark_phase_controlled.py"
EVALUATOR = ROOT / "artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality/scripts/evaluate_policy.py"

VLLM_NVFP4_EXTENSION_DIR = SOURCE_061 / "vllm_nvfp4_extension"
COSPAQ_SPARSE_NVFP4_EXTENSION_DIR = SOURCE_061 / "cospaq_sparse_nvfp4_extension"
VLLM_SPARSE_NVFP4_EXTENSION_DIR = SOURCE_061 / "vllm_sparse_nvfp4_extension"
COSPAQ_SPARSE_BF16_EXTENSION_DIR = SOURCE_061 / "cospaq_sparse_bf16_extension"
VLLM_SPARSE_BF16_EXTENSION_DIR = SOURCE_061 / "vllm_sparse_bf16_extension"

BATCH = 8
INPUT_TOKENS = 2048
MAX_BATCHED_TOKENS = BATCH * INPUT_TOKENS
REPEATS = 5
TARGET_SPEEDUP = 1.7573628343077703
TARGET_ARC_CHALLENGE = 0.5110921501706485
METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "w4a16_ours",
)
KERNEL = {
    "dense_bf16": "dense_bf16",
    "dense_nvfp4": "dense_nvfp4",
    "sparse_bf16": "sparse_bf16",
    "sparse_nvfp4": "sparse_nvfp4",
    "w4a16_ours": "marlin_nvfp4",
}
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
