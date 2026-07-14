#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then echo "usage: $0 LABEL POLICY_JSON GPU BATCH" >&2; exit 2; fi
LABEL="$1"; POLICY="$2"; GPU="$3"; BATCH="$4"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/arc_challenge/full/${LABEL}.json"; mkdir -p "${ROOT}/arc_challenge/full" "${ROOT}/arc_challenge/logs"
[ -s "${OUT}" ] && exit 0
CUDA_VISIBLE_DEVICES="${GPU}" /home/agent/wja/miniconda3/envs/cospaq/bin/python "${ROOT}/scripts/evaluate_arc_challenge.py" --policy-json "${POLICY}" --label "${LABEL}" --output-json "${OUT}" --gpu 0 --batch-size "${BATCH}"
