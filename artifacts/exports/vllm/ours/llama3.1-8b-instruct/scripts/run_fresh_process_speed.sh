#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-/home/agent/wja/project/my/cospaq/test/vllm}"
CUTLASS_ROOT="${CUTLASS_ROOT:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-6}"
WARMUPS="${WARMUPS:-1}"
REPEATS="${REPEATS:-10}"

export PYTHONPATH="${VLLM_ROOT}/vllm:${VLLM_ROOT}:${CUTLASS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0 PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
export PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"

run_one() {
  local output="$1" kind="$2" index="$3"
  local out_dir="${ROOT}/max_speed/prefill_decode/fresh_process_speed/runs"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${VLLM_ROOT}/artifacts/dev/013_phase_hetero_speed_matrix/benchmark_one.py" \
    --model "${ROOT}/max_speed/prefill_decode/checkpoint" --label ours_max_speed --category phase_hetero \
    --scenario prefill_decode --input-len 2048 --output-len "${output}" --batch-size 16 \
    --output-json "${out_dir}/${kind}_o${output}_r${index}.json" --repo-root "${VLLM_ROOT}" \
    --phase-artifact-dir "${VLLM_ROOT}/artifacts/dev/012_phase_hetero_linear" --cutlass-wrapper-path "${CUTLASS_ROOT}" \
    --phase-hetero --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9
}
for output in 1 80; do
  for ((i=0; i<WARMUPS; ++i)); do run_one "${output}" warmup "${i}"; done
  for ((i=0; i<REPEATS; ++i)); do run_one "${output}" measured "${i}"; done
done
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" --root "${ROOT}"
