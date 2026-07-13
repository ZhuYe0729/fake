#!/usr/bin/env bash
# Real 100-block WikiText NLL for one dense-grid policy.
set -euo pipefail

POINT="${1:?dense-grid point index required}"
GPU="${2:?physical GPU index required}"
REPO="/root/wja/project/my/cospaq/fake"
ROOT="${REPO}/artifacts/debug/036_llama2_prefill_decode_intermediate_points"
PROXY="${REPO}/artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy"
POLICY="${ROOT}/prefill_decode/pareto/policies/point_$(printf '%03d' "${POINT}").json"
OUT="${ROOT}/actual_nll/point_${POINT}.csv"
mkdir -p "${ROOT}/actual_nll" "${ROOT}/logs"
[ -f "${OUT}" ] && exit 0
CUDA_VISIBLE_DEVICES="${GPU}" /home/agent/wja/miniconda3/envs/cospaq/bin/python \
  "${PROXY}/scripts/evaluate_wikitext_nll.py" --scenario prefill_decode --policy "dense_grid_${POINT}" \
  --policy-json "${POLICY}" --output-root "${PROXY}" --blocks 100 --gpu 0 --output-csv "${OUT}" \
  > "${ROOT}/logs/nll_point_${POINT}.log" 2>&1
