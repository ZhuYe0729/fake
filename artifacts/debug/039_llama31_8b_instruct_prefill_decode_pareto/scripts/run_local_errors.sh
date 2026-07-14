#!/usr/bin/env bash
# Eight independent phase/method measurements, capped at four concurrent GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
methods=(dense_nvfp4 sparse_bf16 sparse_nvfp4 w4a16_ours); jobs=(); gpu=1
for phase in prefill decode; do for method in "${methods[@]}"; do
  out="${ROOT}/local_errors/${phase}_${method}.csv"; [ -s "${out}" ] && continue
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT}/scripts/collect_local_errors.py" --phase "${phase}" --method "${method}" --gpu 0 > "${ROOT}/logs/local_${phase}_${method}.log" 2>&1 & jobs+=("$!")
  gpu=$((gpu % 4 + 1)); if [ "${#jobs[@]}" -ge 4 ]; then wait "${jobs[0]}"; jobs=("${jobs[@]:1}"); fi
done; done
for job in "${jobs[@]}"; do wait "${job}"; done
