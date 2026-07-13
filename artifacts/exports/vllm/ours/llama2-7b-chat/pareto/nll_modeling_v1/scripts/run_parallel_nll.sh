#!/usr/bin/env bash
# Eight independent HF workers; each writes an isolated CSV shard, avoiding
# concurrent updates of the scenario-level result file.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=, read -r -a gpu_list <<< "${GPUS}"
for scenario in prefill_only prefill_decode; do
  mkdir -p "${ROOT}/nll/shards/${scenario}" "${ROOT}/logs/${scenario}"
  for i in $(seq 0 29); do
    # Keep exactly one 7B model resident per GPU; launching all 30 at once
    # would place four models on each 32-GB card.
    while [ "$(jobs -rp | wc -l)" -ge "${#gpu_list[@]}" ]; do wait -n; done
    printf -v policy 'p%02d' "${i}"; gpu="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    # Use physical ids directly: the elevated runtime does not reliably
    # remap CUDA_VISIBLE_DEVICES for child Python processes.
    "${PYTHON}" "${ROOT}/scripts/evaluate_policy_nll.py" --scenario "${scenario}" --policy "${policy}" --gpu "${gpu}" --batch-size 1 --output-csv "${ROOT}/nll/shards/${scenario}/${policy}.csv" > "${ROOT}/logs/${scenario}/${policy}.log" 2>&1 &
  done
  wait
  "${PYTHON}" "${ROOT}/scripts/merge_nll_shards.py" --scenario "${scenario}"
  "${PYTHON}" "${ROOT}/scripts/fit_quality_model.py" --scenario "${scenario}"
done
