#!/usr/bin/env bash
set -euo pipefail

label=$1
policy=$2
gpu=$3
root=/home/agent/wja/project/my/cospaq/fake/artifacts/debug/037_llama2_prefill_only_pareto
out="$root/arc_challenge/screen/${label}.json"

mkdir -p "$root/arc_challenge/logs"
if [[ -s "$out" ]]; then
  echo "already completed: $label"
  exit 0
fi

cd /home/agent/wja/project/my/cospaq/fake
/home/agent/wja/miniconda3/envs/cospaq/bin/python "$root/scripts/evaluate_arc_challenge.py" \
  --label "$label" --policy-json "$policy" --output-json "$out" \
  --gpu "$gpu" --batch-size 4 --limit 128
