#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
EXPORTER="${EXPORTER:-${ROOT}/../llama2-7b-chat/scripts/export_max_speed_checkpoint.py}"
GPU="${GPU:-5}"

for scenario in prefill_only prefill_decode; do
  out="${ROOT}/max_speed/${scenario}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_max_speed_policy.py" --scenario "${scenario}" --model-path "${MODEL_PATH}" --output-dir "${out}/policy"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${EXPORTER}" --model-path "${MODEL_PATH}" --policy-json "${out}/policy/phase_hetero_policy.json" --output-dir "${out}/checkpoint" --force --prune
done
