#!/usr/bin/env bash
# Usage: run_nll_shard.sh GPU START END. GPU is a physical device ID.
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "usage: $0 GPU START END" >&2; exit 2; fi
GPU="$1"; START="$2"; END="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
REFERENCE="${ROOT}/nll_shards/p00.csv"
mkdir -p "${ROOT}/nll_shards" "${ROOT}/logs"
if [ ! -s "${REFERENCE}" ]; then echo "missing dense reference ${REFERENCE}" >&2; exit 3; fi
for index in $(seq "${START}" "${END}"); do
  policy=$(printf 'p%02d' "${index}")
  output="${ROOT}/nll_shards/${policy}.csv"
  [ -s "${output}" ] && continue
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${ROOT}/scripts/evaluate_wikitext_nll.py" --policy "${policy}" --gpu 0 --blocks 100 --batch-size 1 --dense-reference-json "${REFERENCE}" --output-csv "${output}" > "${ROOT}/logs/nll_${policy}.log" 2>&1
done
