#!/usr/bin/env bash
# Measure 100-block teacher-forced WikiText prefill NLL for one refined policy.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 LABEL POLICY_JSON GPU" >&2
  exit 2
fi

LABEL=$1
POLICY=$2
GPU=$3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${ROOT%/037_llama2_prefill_only_pareto}/033_llama2_7b_chat_wikitext_phase_nll_proxy
OUT=$ROOT/refined_bridge/nll/$LABEL.csv
mkdir -p "$(dirname "$OUT")" "$ROOT/refined_bridge/logs"
[ -f "$OUT" ] && exit 0
/home/agent/wja/miniconda3/envs/cospaq/bin/python "$SOURCE/scripts/evaluate_wikitext_nll.py" \
  --scenario prefill_only --policy "$LABEL" --policy-json "$POLICY" \
  --output-root "$SOURCE" --blocks 100 --gpu "$GPU" --output-csv "$OUT" \
  > "$ROOT/refined_bridge/logs/nll_${LABEL}.log" 2>&1
