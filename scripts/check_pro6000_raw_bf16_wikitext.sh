#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/artifacts/debug/067_llama31_pro6000_prefill_only/config.current.env"
GPU="${1:-0}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing PRO 6000 configuration: ${CONFIG}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
source "${CONFIG}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${COSPAQ_VLLM_ROOT}/vllm:${COSPAQ_VLLM_ROOT}:${REPO_ROOT}/fake/kernels/cutlass/cutlass_wrapper:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${COSPAQ_VLLM_PYTHON}" - <<'PY'
import importlib.metadata
import json
import sys

sys.path.insert(
    0,
    "artifacts/debug/067_llama31_pro6000_prefill_only/scripts",
)

from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.tasks import TaskManager
from vllm_compat import (
    assert_chunked_prefill_disabled,
    force_v1_chunked_prefill_disabled,
)


model_path = "/root/data/models/Meta-Llama-3.1-8B-Instruct"
force_v1_chunked_prefill_disabled()

lm = VLLM(
    pretrained=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    batch_size=1,
    max_model_len=2048,
    tensor_parallel_size=1,
    enforce_eager=True,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    gpu_memory_utilization=0.90,
    max_num_seqs=1,
    skip_mm_profiling=True,
)

try:
    assert_chunked_prefill_disabled(lm)
    result = simple_evaluate(
        model=lm,
        tasks=["wikitext"],
        num_fewshot=0,
        batch_size=1,
        log_samples=False,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
        task_manager=TaskManager(),
    )
    metrics = result["results"]["wikitext"]
    print(
        json.dumps(
            {
                "model": model_path,
                "vllm": importlib.metadata.version("vllm"),
                "transformers": importlib.metadata.version("transformers"),
                "lm_eval": importlib.metadata.version("lm-eval"),
                "word_perplexity": metrics["word_perplexity,none"],
                "byte_perplexity": metrics["byte_perplexity,none"],
                "bits_per_byte": metrics["bits_per_byte,none"],
            },
            indent=2,
        )
    )
finally:
    try:
        lm.model.llm_engine.engine_core.shutdown()
    except Exception:
        pass
PY
