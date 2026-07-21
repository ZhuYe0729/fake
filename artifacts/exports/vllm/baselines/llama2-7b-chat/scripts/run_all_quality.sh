#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../../" && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/data/datasets/flaxquant}"
MODEL_PATH="${MODEL_PATH:-/root/data/models/Llama-2-7b-chat-hf}"
BERTSCORE_MODEL="${BERTSCORE_MODEL:-/root/data/models/bert_score/roberta-large}"
IWSLT_FILTER_TOKENIZER="${IWSLT_FILTER_TOKENIZER:-/root/data/models/lmsys/vicuna-7b-v1.5}"
HF_HOME="${HF_HOME:-/root/data/huggingface}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export LLAMA2_7B_CHAT_MODEL_PATH="${MODEL_PATH}"
SUBSET_DIR="${DATA_ROOT}/cnn_dailymail_3.0.0_test_random1000_seed42"

if [[ ! -f "${SUBSET_DIR}/dataset_dict.json" ]]; then
  echo "Missing CNN/DM subset: ${SUBSET_DIR}" >&2
  exit 1
fi

for method in dense_bf16 dense_nvfp4 sparse_bf16 sparse_nvfp4 marlin_nvfp4; do
  for dataset in cnn_dm_1000 dsum IWSLT; do
    python "${SCRIPT_DIR}/pmpd_vllm_eval.py" \
      --method "${method}" \
      --dataset "${dataset}" \
      --data-root "${DATA_ROOT}" \
      --bertscore-model "${BERTSCORE_MODEL}" \
      --iwslt-filter-tokenizer "${IWSLT_FILTER_TOKENIZER}" \
      "$@"
  done
done
