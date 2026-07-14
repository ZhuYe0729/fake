#!/usr/bin/env bash
# Advance only after every frozen 100-block NLL shard is present and valid.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
while true; do
  count=$(find "${ROOT}/nll_shards" -maxdepth 1 -name 'p*.csv' -type f | wc -l)
  [ "${count}" -eq 72 ] && break
  sleep 60
done
"${PYTHON_BIN}" "${ROOT}/scripts/merge_nll_shards.py"
"${PYTHON_BIN}" "${ROOT}/scripts/fit_quality_proxy.py"
"${PYTHON_BIN}" "${ROOT}/scripts/build_speed_calibration_design.py"
"${PYTHON_BIN}" "${ROOT}/scripts/solve_predicted_pareto.py"
touch "${ROOT}/quality_pipeline_complete"
