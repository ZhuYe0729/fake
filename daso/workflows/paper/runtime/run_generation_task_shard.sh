#!/usr/bin/env bash
# One fresh vLLM process per policy/data shard; batches reuse that process.
set -euo pipefail

: "${DASO_VLLM_ROOT:?set DASO_VLLM_ROOT}"
: "${DASO_CUTLASS_WRAPPER_ROOT:?set DASO_CUTLASS_WRAPPER_ROOT}"
VLLM_ROOT="${DASO_VLLM_ROOT}"
CUTLASS_ROOT="${DASO_CUTLASS_WRAPPER_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:?checkpoint required}"
DATASET="${DATASET:?dataset required}"
QUESTION_BEGIN="${QUESTION_BEGIN:?question begin required}"
QUESTION_END="${QUESTION_END:?question end required}"
OUT_DIR="${OUT_DIR:?output directory required}"
LABEL="${LABEL:?label required}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
: "${IWSLT_FILTER_TOKENIZER:?set IWSLT_FILTER_TOKENIZER}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

mkdir -p "${OUT_DIR}/logs/${DATASET}"
export PYTHONPATH="${VLLM_ROOT}/vllm:${VLLM_ROOT}:${CUTLASS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0
export PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
export PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
export NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
export MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
export SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
export SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"

"${PYTHON_BIN}" "${VLLM_ROOT}/artifacts/dev/011_phase_switch_linear_test/pmpd_vllm_eval.py" \
  --dataset "${DATASET}" --question-begin "${QUESTION_BEGIN}" --question-end "${QUESTION_END}" --batch-size "${BATCH_SIZE}" \
  --model-path "${CHECKPOINT}" --model-id "${LABEL}" --output-dir "${OUT_DIR}" \
  --repo-root "${VLLM_ROOT}" --artifact-dir "${VLLM_ROOT}/artifacts/dev/011_phase_switch_linear_test" \
  --phase-artifact-dir "${VLLM_ROOT}/artifacts/dev/012_phase_hetero_linear" --cutlass-wrapper-path "${CUTLASS_ROOT}" \
  --phase-hetero --max-num-batched-tokens 15360 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --iwslt-filter-tokenizer "${IWSLT_FILTER_TOKENIZER}" --skip-metrics \
  > "${OUT_DIR}/logs/${DATASET}/shard.log" 2>&1
