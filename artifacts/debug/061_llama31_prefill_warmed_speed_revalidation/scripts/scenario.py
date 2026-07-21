"""Immutable assets and protocol for Llama3 prefill E2E speed revalidation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/llama31_8b_instruct"
SOURCE = ROOT / "artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/llama31_8b_instruct"
SOURCE_038 = ROOT / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto"
MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
EXPORT = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/scripts/export_phase_checkpoint.py"
PHASE_EXPORTER = Path("/home/agent/wja/project/my/cospaq/test/vllm/artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py")
REUSE_DENSE_EXPORTER = ROOT / "artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/scripts/export_phase_reuse_dense_nvfp4.py"
CANONICAL = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/canonical/prepared"
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
VLLM_NVFP4_EXTENSION_DIR = EXP.parent / "vllm_nvfp4_extension"
COSPAQ_SPARSE_NVFP4_EXTENSION_DIR = EXP.parent / "cospaq_sparse_nvfp4_extension"
VLLM_SPARSE_NVFP4_EXTENSION_DIR = EXP.parent / "vllm_sparse_nvfp4_extension"
COSPAQ_SPARSE_BF16_EXTENSION_DIR = EXP.parent / "cospaq_sparse_bf16_extension"
VLLM_SPARSE_BF16_EXTENSION_DIR = EXP.parent / "vllm_sparse_bf16_extension"
BENCH = ROOT / "artifacts/debug/059_llama31_prefill_speed_decomposition/scripts/benchmark_phase_controlled.py"
VLLM_PYTHON = Path("/home/agent/wja/miniconda3/envs/vllm/bin/python")
BATCH, INPUT_TOKENS, REPEATS = 8, 2048, 5
MAX_BATCHED_TOKENS = BATCH * INPUT_TOKENS
METHODS = ("dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "w4a16_ours")
KERNEL = {"dense_bf16": "dense_bf16", "dense_nvfp4": "dense_nvfp4", "sparse_bf16": "sparse_bf16", "sparse_nvfp4": "sparse_nvfp4", "w4a16_ours": "marlin_nvfp4"}
TYPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
