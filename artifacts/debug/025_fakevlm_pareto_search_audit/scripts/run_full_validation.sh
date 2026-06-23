#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/debug/025_fakevlm_pareto_search_audit"

cd "${PROJECT_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-025-search-audit}"
python "${OUTPUT_ROOT}/scripts/make_subset.py" --output-root "${OUTPUT_ROOT}"
python "${OUTPUT_ROOT}/scripts/generate_search_policies.py" --output-root "${OUTPUT_ROOT}"
python "${OUTPUT_ROOT}/scripts/launch_validation.py" --output-root "${OUTPUT_ROOT}" --gpus "${GPUS:-0,1,2,3,4,5}" "$@"
python "${OUTPUT_ROOT}/scripts/summarize_search.py" --output-root "${OUTPUT_ROOT}"
