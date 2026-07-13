#!/usr/bin/env bash
# One GPU, one policy: real 100-block WikiText phase NLL.
set -euo pipefail
POINT="${1:?point}"; GPU="${2:?gpu}"
REPO="/root/wja/project/my/cospaq/fake"
SOURCE="${REPO}/artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver"
PROXY="${REPO}/artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy"
ROOT="${REPO}/artifacts/debug/035_llama2_prefill_decode_e2e_speed_model"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
POLICY="${SOURCE}/prefill_decode/pareto/policies/point_$(printf '%03d' "${POINT}").json"
OUT="${ROOT}/actual_nll/point_${POINT}.csv"
mkdir -p "${ROOT}/actual_nll" "${ROOT}/logs"
[ -f "${OUT}" ] && exit 0
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${PROXY}/scripts/evaluate_wikitext_nll.py" \
  --scenario prefill_decode --policy "point_${POINT}" --policy-json "${POLICY}" \
  --output-root "${PROXY}" --blocks 100 --gpu 0 --output-csv "${OUT}" \
  > "${ROOT}/logs/nll_point_${POINT}.log" 2>&1
