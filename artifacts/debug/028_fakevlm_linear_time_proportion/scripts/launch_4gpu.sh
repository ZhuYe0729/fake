#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/028_fakevlm_linear_time_proportion}"
PYTHON="${PYTHON:-python}"
CONDA_ENV="${CONDA_ENV:-cospaq}"
GPUS=(${GPUS:-7 6 5 4})
WORKLOADS=(${WORKLOADS:-prefill_b1_i1024 prefill_b4_i1024 prefill_b16_i1024 prefill_b4_i4096 normal_01 normal_02})
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
WORKERS="${WORKERS:-1}"
OVERWRITE="${OVERWRITE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${ROOT}/logs"
echo "Repo root: ${REPO_ROOT}"
echo "Output root: ${ROOT}"
echo "GPUs: ${GPUS[*]}"
echo "Workloads: ${WORKLOADS[*]}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo

declare -A GPU_PIDS=()
declare -A GPU_WORKLOADS=()

launch_task() {
  local gpu="$1"
  local workload="$2"
  local log="${ROOT}/logs/${workload}_gpu${gpu}.log"
  local -a cmd=(
    "${PYTHON}" "${ROOT}/scripts/run_linear_proportion.py"
    --output-root "${ROOT}"
    --workload "${workload}"
    --gpu 0
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --workers "${WORKERS}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  echo "[launch] gpu=${gpu} workload=${workload} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log}" 2>&1 &
  GPU_PIDS["${gpu}"]=$!
  GPU_WORKLOADS["${gpu}"]="${workload}"
}

wait_one_gpu() {
  while true; do
    for gpu in "${GPUS[@]}"; do
      local pid="${GPU_PIDS[${gpu}]:-}"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        if wait "${pid}"; then
          echo "[done] gpu=${gpu} workload=${GPU_WORKLOADS[${gpu}]}"
        else
          echo "[fail] gpu=${gpu} workload=${GPU_WORKLOADS[${gpu}]}" >&2
          exit 1
        fi
        unset "GPU_PIDS[${gpu}]"
        unset "GPU_WORKLOADS[${gpu}]"
        return
      fi
    done
    sleep 2
  done
}

task_index=0
while [[ "${task_index}" -lt "${#WORKLOADS[@]}" ]]; do
  launched=0
  for gpu in "${GPUS[@]}"; do
    if [[ -z "${GPU_PIDS[${gpu}]:-}" && "${task_index}" -lt "${#WORKLOADS[@]}" ]]; then
      launch_task "${gpu}" "${WORKLOADS[${task_index}]}"
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
      echo "[done] gpu=${gpu} workload=${GPU_WORKLOADS[${gpu}]}"
    else
      echo "[fail] gpu=${gpu} workload=${GPU_WORKLOADS[${gpu}]}" >&2
      exit 1
    fi
  fi
done

"${PYTHON}" "${ROOT}/scripts/summarize.py" --output-root "${ROOT}"
echo "Done. Summary: ${ROOT}/summary/analysis_report.md"
