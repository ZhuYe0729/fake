"""Frozen paths and protocol for debug 069."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/069_llama31_pareto_search_audit"
MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")

SOURCE_038 = ROOT / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto"
SOURCE_057 = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto"
EXP_057 = SOURCE_057 / "llama31_8b_instruct"
SOURCE_060 = ROOT / "artifacts/debug/060_two_model_two_scenario_result_consolidation"
SOURCE_061 = ROOT / "artifacts/debug/061_llama31_prefill_warmed_speed_revalidation"
EXP_061 = SOURCE_061 / "llama31_8b_instruct"
SOURCE_068 = ROOT / "artifacts/debug/068_llama31_prefill_uniform_coverage_refinement"
EXP_068 = SOURCE_068 / "llama31_8b_instruct"

PREFILL = DEBUG / "llama31_8b_instruct/prefill_only"
DECODE = DEBUG / "llama31_8b_instruct/prefill_decode"
CANONICAL = EXP_057 / "canonical/prepared"

COSPAQ_PYTHON = Path("/home/agent/wja/miniconda3/envs/cospaq/bin/python")
VLLM_PYTHON = Path("/home/agent/wja/miniconda3/envs/vllm/bin/python")
VLLM_ROOT = Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
MARLIN_EXTENSION = CUTLASS / "artifacts/torch_extensions_marlin_nvfp4"
RUNTIME_TMP = DEBUG / "temporary/runtime_tmp"
PREFILL_EXPORTER = SOURCE_061 / "scripts/export_phase_reuse_dense_nvfp4.py"
PREFILL_BENCH = ROOT / "artifacts/debug/059_llama31_prefill_speed_decomposition/scripts/benchmark_phase_controlled.py"
PREFILL_EVALUATOR = ROOT / "artifacts/debug/068_llama31_prefill_uniform_coverage_refinement/scripts/evaluate_arc_challenge.py"
DECODE_EXPORTER = SOURCE_057 / "scripts/export_phase_checkpoint.py"
DECODE_SPEED_RUNNER = ROOT / "artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/run_speed_policy.sh"
DECODE_VERIFIER = ROOT / "artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/verify_canonical_checkpoint.py"
TASK_SHARD_RUNNER = ROOT / "artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/scripts/run_task_quality_shard.sh"
PMPD_EVAL = ROOT / "references/pmpd_eval_kit/pmpd_eval.py"

VLLM_NVFP4_EXTENSION = SOURCE_061 / "vllm_nvfp4_extension"
COSPAQ_SPARSE_NVFP4_EXTENSION = SOURCE_061 / "cospaq_sparse_nvfp4_extension"
VLLM_SPARSE_NVFP4_EXTENSION = SOURCE_061 / "vllm_sparse_nvfp4_extension"
COSPAQ_SPARSE_BF16_EXTENSION = SOURCE_061 / "cospaq_sparse_bf16_extension"
VLLM_SPARSE_BF16_EXTENSION = SOURCE_061 / "vllm_sparse_bf16_extension"

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
SEED = 20260723

PREFILL_PARENTS = ("point_000", "point_004", "point_006", "fine_q208", "point_014")
DECODE_PARENTS = ("point_001", "point_004", "point_006", "point_009", "point_011")
RANDOM_RATIOS = (
    0.05, 0.05, 0.10, 0.10, 0.20, 0.20, 0.35, 0.35,
    0.50, 0.50, 0.65, 0.65, 0.80, 0.80, 1.00,
)
NEIGHBORHOOD_RATES = (0.05, 0.15)
SUSPICIOUS_COUNT = 32

PREFILL_ANCHOR_MS = 1087.526896968484
DECODE_ANCHOR_MS = 2198.6881513148546
DECODE_STEPS = 63


def parent_path(scenario: str, policy_id: str) -> Path:
    if scenario == "prefill_only":
        if policy_id == "fine_q208":
            return EXP_068 / "policies/fine_q208.json"
        return EXP_061 / f"pareto/policies/{policy_id}.json"
    return EXP_057 / f"pareto/policies/{policy_id}.json"


def candidate_path(root: Path, policy_id: str) -> Path:
    for family in ("random", "neighborhood", "suspicious"):
        path = root / f"policies/{family}/{policy_id}.json"
        if path.exists():
            return path
    raise FileNotFoundError(policy_id)
