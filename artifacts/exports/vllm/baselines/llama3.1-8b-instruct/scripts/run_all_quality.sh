#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../" && pwd)"
DATA_ROOT="/home/agent/wja/data/datasets/flaxquant"
SUBSET_DIR="${DATA_ROOT}/cnn_dailymail_3.0.0_test_random1000_seed42"

if [[ ! -d "${SUBSET_DIR}" ]]; then
  python "${REPO_ROOT}/references/pmpd_eval_kit/make_cnn_dm_subset.py" \
    --data-root "${DATA_ROOT}"
fi

for method in dense_bf16 dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4; do
  for dataset in cnn_dm_1000 dsum IWSLT; do
    python "${SCRIPT_DIR}/pmpd_vllm_eval.py" \
      --method "${method}" \
      --dataset "${dataset}" \
      --resume \
      "$@"
  done
done
