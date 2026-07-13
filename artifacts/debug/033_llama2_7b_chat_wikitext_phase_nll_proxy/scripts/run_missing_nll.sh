#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SCENARIO="${1:-all}"
BLOCKS="${BLOCKS:-100}"
IFS=, read -r -a lanes <<< "${GPUS}"
run() {
  local scenario="$1"; local -a pids=()
  mkdir -p "${ROOT}/nll/shards/${scenario}" "${ROOT}/logs/${scenario}"
  for lane in "${!lanes[@]}"; do
    (
      for ((i=lane;i<72;i+=${#lanes[@]})); do
        printf -v policy 'p%02d' "$i"; shard="${ROOT}/nll/shards/${scenario}/${policy}.csv"
        [ -s "$shard" ] && grep -q ",${BLOCKS}," "$shard" && continue
        "${PYTHON}" "${ROOT}/scripts/evaluate_wikitext_nll.py" --scenario "$scenario" --policy "$policy" --gpu "${lanes[$lane]}" --blocks "$BLOCKS" --output-csv "$shard" > "${ROOT}/logs/${scenario}/${policy}.log" 2>&1
      done
    ) & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "${PYTHON}" "${ROOT}/scripts/merge_shards.py" --scenario "$scenario" --blocks "$BLOCKS"
  "${PYTHON}" "${ROOT}/scripts/fit_ablations.py" --scenario "$scenario"
}
case "$SCENARIO" in prefill_only|prefill_decode) run "$SCENARIO";;all) run prefill_only;run prefill_decode;;*) exit 2;;esac
