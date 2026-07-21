#!/usr/bin/env bash
# True phase-switch canonical NLL closure for one solved Pareto policy.
set -euo pipefail

if [ "$#" -ne 2 ]; then echo "usage: $0 POLICY_ID PHYSICAL_GPU" >&2; exit 2; fi
POLICY_ID="$1"
GPU="$2"
BLOCKS="${BLOCKS:-100}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="/root/wja/project/my/cospaq/fake"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
MODEL="/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"
STREAM="${REPO}/artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/stream_phase_policy_nll.py"
CANONICAL="${REPO}/artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/canonical/prepared"
POLICY_DIR="${POLICY_DIR:-${ROOT}/llama2_7b_chat/pareto/policies}"
POLICY="${POLICY_DIR}/${POLICY_ID}.json"
OUT="${ROOT}/llama2_7b_chat/validation/nll/${POLICY_ID}.json"

mkdir -p "$(dirname "${OUT}")" "${ROOT}/logs/pareto_nll"
[ -f "${OUT}" ] && exit 0
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${STREAM}" --model-path "${MODEL}" --tokenizer "${MODEL}" \
  --policy-json "${POLICY}" --samples "${ROOT}/llama2_7b_chat/samples/wikitext_2048_80.pt" \
  --output "${OUT}" --label "${POLICY_ID}" --blocks "${BLOCKS}" \
  --canonical-sparse-bf16-state "${CANONICAL}/sparse_bf16/model.pt" \
  --canonical-sparse-nvfp4-state "${CANONICAL}/sparse_nvfp4/model.pt" \
  > "${ROOT}/logs/pareto_nll/${POLICY_ID}.log" 2>&1
