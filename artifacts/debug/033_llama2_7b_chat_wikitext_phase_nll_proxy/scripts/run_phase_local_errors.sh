#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
METHODS=(dense_nvfp4 sparse_bf16 sparse_nvfp4 w4a16_ours)
PHASES=(prefill decode)
# Each worker holds a 13-GB prepared state in host RAM. Two workers are safe
# on this host; eight workers are not, even though eight GPUs are available.
for start in 0 2 4 6; do
  pids=()
  for index in "$start" "$((start+1))"; do
    phase="${PHASES[$((index/4))]}"; method="${METHODS[$((index%4))]}"
    "${PYTHON}" "${ROOT}/scripts/collect_phase_local_errors.py" --phase "$phase" --method "$method" --gpu "$index" > "${ROOT}/logs/local_${phase}_${method}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
