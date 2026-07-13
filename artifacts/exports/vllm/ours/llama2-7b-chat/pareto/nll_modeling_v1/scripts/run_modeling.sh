#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
GPU="${GPU:-0}"

"${PYTHON}" "${ROOT}/scripts/generate_calibration.py"
for scenario in prefill_only prefill_decode; do
  "${PYTHON}" "${ROOT}/scripts/predict_policy_speed.py" --scenario "${scenario}"
  "${PYTHON}" "${ROOT}/scripts/evaluate_policy_nll.py" --scenario "${scenario}" --gpu "${GPU}"
  "${PYTHON}" "${ROOT}/scripts/fit_quality_model.py" --scenario "${scenario}"
done
