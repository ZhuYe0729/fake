#!/usr/bin/env bash
# Evaluate one frozen policy's 100-block WikiText prefill NLL on one GPU.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 POINT GPU" >&2
  exit 2
fi

POINT="$1"
GPU="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD="${ROOT%/037_llama2_prefill_only_pareto}/034_llama2_7b_chat_wikitext_pareto_solver"
SOURCE="${ROOT%/037_llama2_prefill_only_pareto}/033_llama2_7b_chat_wikitext_phase_nll_proxy"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
POLICY="${OLD}/prefill_only/pareto/policies/point_$(printf '%03d' "${POINT}").json"
OUTPUT="${ROOT}/actual_nll/point_${POINT}.csv"
LOG="${ROOT}/logs/nll_point_${POINT}.log"

mkdir -p "${ROOT}/actual_nll" "${ROOT}/logs"
[ -f "${OUTPUT}" ] && exit 0
"${PYTHON}" "${SOURCE}/scripts/evaluate_wikitext_nll.py" \
  --scenario prefill_only --policy "point_${POINT}" --policy-json "${POLICY}" \
  --output-root "${SOURCE}" --blocks 100 --gpu "${GPU}" --output-csv "${OUTPUT}" \
  > "${LOG}" 2>&1
