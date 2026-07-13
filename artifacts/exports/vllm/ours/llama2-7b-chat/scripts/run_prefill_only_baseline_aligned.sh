#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-7}"
OUT_DIR="${ROOT}/max_speed/prefill_only/baseline_aligned_speed"
CHECKPOINT="${ROOT}/max_speed/prefill_only/checkpoint"
mkdir -p "${OUT_DIR}"

run() {
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_phase_baseline_one.py" --checkpoint "${CHECKPOINT}" --output-json "$1"
}

run "${OUT_DIR}/warmup_0.json"
for index in 0 1 2 3 4; do run "${OUT_DIR}/measured_${index}.json"; done
"${PYTHON_BIN}" - <<PY
import json, statistics
from pathlib import Path
root = Path("${OUT_DIR}")
values = [json.loads((root / f"measured_{i}.json").read_text())["elapsed_ms"] for i in range(5)]
(root / "summary.json").write_text(json.dumps({"repeats": 5, "mean_ms": statistics.mean(values), "median_ms": statistics.median(values), "values_ms": values}, indent=2) + "\n")
print((root / "summary.json").read_text())
PY
