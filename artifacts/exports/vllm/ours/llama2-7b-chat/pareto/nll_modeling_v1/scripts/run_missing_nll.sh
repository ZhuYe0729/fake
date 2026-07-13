#!/usr/bin/env bash
# Resume NLL calibration with one serial worker assigned to each physical GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SCENARIO="${1:-all}"
IFS=, read -r -a gpu_list <<< "${GPUS}"

run_scenario() {
  local scenario="$1" lane gpu pid failed=0
  mkdir -p "${ROOT}/nll/shards/${scenario}" "${ROOT}/logs/${scenario}"
  for lane in "${!gpu_list[@]}"; do
    gpu="${gpu_list[$lane]}"
    (
      for ((i=lane; i<30; i+=${#gpu_list[@]})); do
        printf -v policy 'p%02d' "${i}"
        shard="${ROOT}/nll/shards/${scenario}/${policy}.csv"
        if [ -s "${shard}" ]; then
          echo "skip ${scenario}/${policy}: existing shard"
          continue
        fi
        echo "start ${scenario}/${policy} on gpu=${gpu}"
        "${PYTHON}" "${ROOT}/scripts/evaluate_policy_nll.py" --scenario "${scenario}" --policy "${policy}" --gpu "${gpu}" --batch-size 1 --output-csv "${shard}" > "${ROOT}/logs/${scenario}/${policy}.log" 2>&1
        echo "done ${scenario}/${policy} on gpu=${gpu}"
      done
    ) &
    pids[$lane]=$!
  done
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  if [ "${failed}" -ne 0 ]; then
    echo "${scenario}: at least one lane failed; preserve shards and rerun this script" >&2
    return 1
  fi
  "${PYTHON}" "${ROOT}/scripts/merge_nll_shards.py" --scenario "${scenario}"
  "${PYTHON}" "${ROOT}/scripts/fit_quality_model.py" --scenario "${scenario}"
}

case "${SCENARIO}" in
  prefill_only|prefill_decode) run_scenario "${SCENARIO}" ;;
  all) run_scenario prefill_only; run_scenario prefill_decode ;;
  *) echo "usage: $0 [prefill_only|prefill_decode|all]" >&2; exit 2 ;;
esac
