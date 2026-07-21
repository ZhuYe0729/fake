#!/usr/bin/env bash
# Complete only the missing primary task metrics for uniform p03 and p04.
set -euo pipefail

ROOT="/home/agent/wja/project/my/cospaq/fake/artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto"
RUNNER="${ROOT}/scripts/run_uniform_task_quality_shard.sh"
GPU="${1:?GPU required}"
shift

run_shard() {
  local policy="$1" dataset="$2" begin="$3" end="$4"
  local checkpoint="${ROOT}/llama2_7b_chat/speed/runs/${policy}/checkpoint"
  local output="${ROOT}/llama2_7b_chat/task_quality/shards/${policy}/${dataset}/shard_$(printf '%04d' "${begin}")_$(printf '%04d' "${end}")"
  CUDA_VISIBLE_DEVICES="${GPU}" CHECKPOINT="${checkpoint}" DATASET="${dataset}" \
    QUESTION_BEGIN="${begin}" QUESTION_END="${end}" OUT_DIR="${output}" \
    LABEL="uniform_${policy}_prefill_decode" BATCH_SIZE=4 GPU_MEMORY_UTILIZATION=0.75 \
    MAX_MODEL_LEN=4096 "${RUNNER}"
}

while (( "$#" )); do
  run_shard "$1" "$2" "$3" "$4"
  shift 4
  # vLLM/NCCL may retain CUDA memory briefly after process exit.  Without a
  # gap, the next fresh engine can reject its memory-utilization request.
  if (( "$#" )); then
    sleep 20
  fi
done
