#!/usr/bin/env bash
set -euo pipefail

# One fresh Python process per PMPD batch, matching the validated phase-hetero lifecycle.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-/home/agent/wja/project/my/cospaq/test/vllm}"
CUTLASS_ROOT="${CUTLASS_ROOT:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
SCENARIO="${SCENARIO:?set SCENARIO to prefill_only or prefill_decode}"
DATASET="${DATASET:?set DATASET to cnn_dm_1000, dsum, or IWSLT}"
QUESTION_BEGIN="${QUESTION_BEGIN:-0}"
QUESTION_END="${QUESTION_END:?set QUESTION_END to the exclusive dataset sample index to evaluate}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
IWSLT_FILTER_TOKENIZER="${IWSLT_FILTER_TOKENIZER:-/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf}"

CHECKPOINT="${ROOT}/max_speed/${SCENARIO}/checkpoint"
OUT_DIR="${OUT_DIR:-${ROOT}/max_speed/${SCENARIO}/results/quality}"
LOG_DIR="${OUT_DIR}/logs/${DATASET}"
LABEL="${LABEL:-ours_max_speed_${SCENARIO}}"
SKIP_FINAL_METRICS="${SKIP_FINAL_METRICS:-0}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
export PYTHONPATH="${VLLM_ROOT}/vllm:${VLLM_ROOT}:${CUTLASS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0 PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
export PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"

for ((begin=QUESTION_BEGIN; begin<QUESTION_END; begin+=BATCH_SIZE)); do
  end=$((begin + BATCH_SIZE)); (( end > QUESTION_END )) && end="${QUESTION_END}"
  extra=(); (( begin > 0 )) && extra=(--append)
  "${PYTHON_BIN}" "${VLLM_ROOT}/artifacts/dev/011_phase_switch_linear_test/pmpd_vllm_eval.py" \
    --dataset "${DATASET}" --question-begin "${begin}" --question-end "${end}" --batch-size "${BATCH_SIZE}" \
    --model-path "${CHECKPOINT}" --model-id "${LABEL}" --output-dir "${OUT_DIR}" \
    --repo-root "${VLLM_ROOT}" --artifact-dir "${VLLM_ROOT}/artifacts/dev/011_phase_switch_linear_test" \
    --phase-artifact-dir "${VLLM_ROOT}/artifacts/dev/012_phase_hetero_linear" --cutlass-wrapper-path "${CUTLASS_ROOT}" \
    --phase-hetero --max-num-batched-tokens 15360 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --iwslt-filter-tokenizer "${IWSLT_FILTER_TOKENIZER}" --skip-metrics "${extra[@]}" > "${LOG_DIR}/${begin}_${end}.log" 2>&1
done
if [[ "${SKIP_FINAL_METRICS}" != "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/../../../../../references/pmpd_eval_kit/pmpd_eval.py" --dataset "${DATASET}" --metrics-only "${OUT_DIR}/${DATASET}/${LABEL}-fp16.jsonl" > "${LOG_DIR}/metrics.log" 2>&1
fi
