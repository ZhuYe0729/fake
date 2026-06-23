#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STUDY="${SCRIPT_DIR}/run_study.py"
ANALYZE="${SCRIPT_DIR}/analyze.py"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate cospaq
fi

GPUS="${GPUS:-7,6,5,4}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"

COMMON_ARGS=(
  --num-shards "${NUM_SHARDS}"
)

if [[ -n "${LLAMA2_MODEL_PATH:-}" ]]; then
  COMMON_ARGS+=(--model-path "${LLAMA2_MODEL_PATH}")
fi

echo "Repo root: ${REPO_ROOT}"
echo "GPUs: ${GPUS}"
echo "Num shards: ${NUM_SHARDS}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo

run_phase() {
  local phase="$1"
  echo "=== Starting phase: ${phase} ==="
  local pids=()
  for shard in "${!GPU_ARRAY[@]}"; do
    local gpu="${GPU_ARRAY[$shard]}"
    local log_file="${LOG_DIR}/${phase}_gpu${gpu}_shard${shard}.log"
    echo "Launching ${phase}: gpu=${gpu} shard=${shard}/${NUM_SHARDS}, log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${RUN_STUDY}" \
      --phase "${phase}" \
      --gpu 0 \
      --shard-index "${shard}" \
      "${COMMON_ARGS[@]}" \
      > "${log_file}" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "Phase ${phase} had failed worker(s). Check ${LOG_DIR}."
    return 1
  fi
  echo "=== Completed phase: ${phase} ==="
  echo
}

run_phase speed
run_phase breakdown

python "${ANALYZE}"

echo "Done. Summary written under ${SCRIPT_DIR}/summary"
