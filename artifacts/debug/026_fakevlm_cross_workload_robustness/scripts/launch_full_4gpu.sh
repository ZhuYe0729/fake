#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/026_fakevlm_cross_workload_robustness}"
PYTHON="${PYTHON:-python}"
CONDA_ENV="${CONDA_ENV:-cospaq}"
GPUS=(${GPUS:-0 1 2 3})
SCENARIOS=(${SCENARIOS:-prefill_only normal_01 normal_02})
METHODS=(${METHODS:-dense_bf16 uniform_dense_nvfp4 uniform_sparse_bf16 uniform_sparse_nvfp4 uniform_marlin_weight_only uniform_dense_nvfp4_prefill_marlin_decode our_linear_hybrid})
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
WORKERS="${WORKERS:-1}"
OVERWRITE="${OVERWRITE:-0}"
OVERRIDE_OUTPUT_TOKENS="${OVERRIDE_OUTPUT_TOKENS:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${ROOT}/logs"

declare -a TASKS=()
for scenario in "${SCENARIOS[@]}"; do
  for method in "${METHODS[@]}"; do
    TASKS+=("${scenario}:${method}")
  done
done

declare -A GPU_PIDS=()
declare -A GPU_TASKS=()

launch_task() {
  local gpu="$1"
  local task="$2"
  local scenario="${task%%:*}"
  local method="${task##*:}"
  local log="${ROOT}/logs/${scenario}_${method}_gpu${gpu}.log"
  local -a cmd=(
    "${PYTHON}" "${ROOT}/scripts/run_e2e_speed.py"
    --output-root "${ROOT}"
    --scenario "${scenario}"
    --method "${method}"
    --gpu 0
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --calib-samples "${CALIB_SAMPLES}"
    --workers "${WORKERS}"
  )
  if [[ -n "${SAMPLE_LIMIT}" ]]; then
    cmd+=(--sample-limit "${SAMPLE_LIMIT}")
  fi
  if [[ -n "${OVERRIDE_OUTPUT_TOKENS}" ]]; then
    cmd+=(--override-output-tokens "${OVERRIDE_OUTPUT_TOKENS}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  echo "[launch] gpu=${gpu} task=${task} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log}" 2>&1 &
  GPU_PIDS["${gpu}"]=$!
  GPU_TASKS["${gpu}"]="${task}"
}

wait_one_gpu() {
  while true; do
    for gpu in "${GPUS[@]}"; do
      local pid="${GPU_PIDS[${gpu}]:-}"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        if wait "${pid}"; then
          echo "[done] gpu=${gpu} task=${GPU_TASKS[${gpu}]}"
        else
          echo "[fail] gpu=${gpu} task=${GPU_TASKS[${gpu}]}" >&2
          exit 1
        fi
        unset "GPU_PIDS[${gpu}]"
        unset "GPU_TASKS[${gpu}]"
        return
      fi
    done
    sleep 2
  done
}

task_index=0
while [[ "${task_index}" -lt "${#TASKS[@]}" ]]; do
  launched=0
  for gpu in "${GPUS[@]}"; do
    if [[ -z "${GPU_PIDS[${gpu}]:-}" && "${task_index}" -lt "${#TASKS[@]}" ]]; then
      launch_task "${gpu}" "${TASKS[${task_index}]}"
      task_index=$((task_index + 1))
      launched=1
    fi
  done
  if [[ "${launched}" == "0" ]]; then
    wait_one_gpu
  fi
done

for gpu in "${GPUS[@]}"; do
  if [[ -n "${GPU_PIDS[${gpu}]:-}" ]]; then
    pid="${GPU_PIDS[${gpu}]}"
    if wait "${pid}"; then
      echo "[done] gpu=${gpu} task=${GPU_TASKS[${gpu}]}"
    else
      echo "[fail] gpu=${gpu} task=${GPU_TASKS[${gpu}]}" >&2
      exit 1
    fi
  fi
done

"${PYTHON}" "${ROOT}/scripts/summarize_cross_workload.py" --output-root "${ROOT}"
