#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-/home/agent/wja/project/my/cospaq/test/vllm}"
CUTLASS_ROOT="${CUTLASS_ROOT:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-3}"
WARMUPS="${WARMUPS:-1}"
REPEATS="${REPEATS:-10}"

export PYTHONPATH="${VLLM_ROOT}/vllm:${VLLM_ROOT}:${CUTLASS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0 PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
export PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
export NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
export MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
export SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
export SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"

run_one() {
  local scenario="$1" input_len="$2" output_len="$3" batch="$4" checkpoint="$5" kind="$6" index="$7" measured_output="$8"
  local output_dir="${ROOT}/max_speed/${scenario}/fresh_process_speed/runs"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${VLLM_ROOT}/artifacts/dev/013_phase_hetero_speed_matrix/benchmark_one.py" \
    --model "${checkpoint}" --label ours_max_speed --category phase_hetero \
    --scenario "${scenario}" --input-len "${input_len}" --output-len "${measured_output}" --batch-size "${batch}" \
    --output-json "${output_dir}/${kind}_o${measured_output}_r${index}.json" \
    --repo-root "${VLLM_ROOT}" --phase-artifact-dir "${VLLM_ROOT}/artifacts/dev/012_phase_hetero_linear" \
    --cutlass-wrapper-path "${CUTLASS_ROOT}" --phase-hetero --max-num-batched-tokens "$((batch * input_len))" --gpu-memory-utilization 0.9
}

run_scenario() {
  local scenario="$1" input_len="$2" output_len="$3" batch="$4"
  local checkpoint="${ROOT}/max_speed/${scenario}/checkpoint"
  for measured_output in 1 "${output_len}"; do
    for ((index=0; index<WARMUPS; ++index)); do run_one "${scenario}" "${input_len}" "${output_len}" "${batch}" "${checkpoint}" warmup "${index}" "${measured_output}"; done
    for ((index=0; index<REPEATS; ++index)); do run_one "${scenario}" "${input_len}" "${output_len}" "${batch}" "${checkpoint}" measured "${index}" "${measured_output}"; done
  done
}

run_scenario prefill_only 2048 1 8
run_scenario prefill_decode 2048 80 16
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_fresh_process_speed.py" --root "${ROOT}"
