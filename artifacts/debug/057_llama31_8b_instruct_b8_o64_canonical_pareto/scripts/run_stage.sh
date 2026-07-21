#!/usr/bin/env bash
# Invoke proven 056 workflow stages with the independent 057 paths above.
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "usage: $0 SCRIPT [ARGS...]" >&2; exit 2; fi
source "$(dirname "$0")/env.sh"
WORKFLOW="/root/wja/project/my/cospaq/fake/artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts"
exec /home/agent/wja/miniconda3/envs/vllm/bin/python "$WORKFLOW/$1" "${@:2}"
