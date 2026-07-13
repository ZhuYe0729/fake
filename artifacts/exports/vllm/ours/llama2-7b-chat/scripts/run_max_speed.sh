#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"
source /home/agent/wja/miniconda3/etc/profile.d/conda.sh

for scenario in prefill_only prefill_decode; do
  SCENARIO_ROOT="${ROOT}/max_speed/${scenario}"
  POLICY_DIR="${SCENARIO_ROOT}/policy"
  CHECKPOINT="${SCENARIO_ROOT}/checkpoint"
  conda activate cospaq
  python "${SCRIPT_DIR}/generate_max_speed_policy.py" --scenario "${scenario}" --model-path "${MODEL_PATH}" --output-dir "${POLICY_DIR}"
  python "${SCRIPT_DIR}/export_max_speed_checkpoint.py" --policy-json "${POLICY_DIR}/phase_hetero_policy.json" --output-dir "${CHECKPOINT}" --model-path "${MODEL_PATH}" --force --prune
  conda activate vllm
  python "${SCRIPT_DIR}/benchmark_phase_hetero.py" --scenario "${scenario}" --checkpoint "${CHECKPOINT}" --output-dir "${SCENARIO_ROOT}/results/speed"
  for dataset in cnn_dm_1000 dsum IWSLT; do
    python "${SCRIPT_DIR}/eval_phase_hetero_quality.py" --checkpoint "${CHECKPOINT}" --dataset "${dataset}" --output-dir "${SCENARIO_ROOT}/results/quality"
  done
done
python "${SCRIPT_DIR}/summarize_results.py" --root "${ROOT}"
