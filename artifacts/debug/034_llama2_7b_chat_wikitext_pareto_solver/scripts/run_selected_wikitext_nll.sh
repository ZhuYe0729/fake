#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${SOURCE:-/root/wja/project/my/cospaq/fake/artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy}"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
EVAL="${SOURCE}/scripts/evaluate_wikitext_nll.py"
for scenario in prefill_only prefill_decode; do
  mapfile -t rows < <("${PYTHON}" -c "import json;[print(x['point_index']+' '+x['policy_json']) for x in json.load(open('${ROOT}/validation/${scenario}/selection.json'))]")
  pids=(); gpu=0
  for row in "${rows[@]}"; do
    point="${row%% *}"; policy="${row#* }"; out="${ROOT}/validation/${scenario}/nll_point_${point}.csv"
    "${PYTHON}" "${EVAL}" --scenario "${scenario}" --policy "point_${point}" --policy-json "${policy}" --output-root "${SOURCE}" --blocks 100 --gpu "${gpu}" --output-csv "${out}" > "${ROOT}/validation/${scenario}/nll_point_${point}.log" 2>&1 &
    pids+=("$!"); gpu=$((gpu+1))
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done
