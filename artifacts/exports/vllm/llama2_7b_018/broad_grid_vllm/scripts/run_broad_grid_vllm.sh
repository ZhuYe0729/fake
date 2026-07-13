#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../../../../.." && pwd)"
PYTHON="${PYTHON:-/root/wja/miniconda3/envs/vllm/bin/python}"
GPUS="${GPUS:-1,2,3,4,5,6}"
CUTLASS_WRAPPER="${CUTLASS_WRAPPER:-${REPO_ROOT}/fake/kernels/cutlass/cutlass_wrapper}"

unset _CUDA_COMPAT_STATUS
export NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_WRAPPER}"
export SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_WRAPPER}"
export SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_WRAPPER}"
export MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_WRAPPER}"

"${PYTHON}" "${ROOT}/scripts/benchmark_broad_grid_vllm_parallel.py" \
  --python "${PYTHON}" \
  --output-dir "${ROOT}/results" \
  --gpus "${GPUS}" \
  --warmup-iters "${WARMUP_ITERS:-1}" \
  --iters "${ITERS:-3}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --max-total-prompt-tokens "${MAX_TOTAL_PROMPT_TOKENS:-131072}" \
  --continue-on-error

"${PYTHON}" "${ROOT}/scripts/summarize_broad_grid_vllm.py" \
  --summary-long "${ROOT}/results/summary_long.csv" \
  --output-dir "${ROOT}/summary"
