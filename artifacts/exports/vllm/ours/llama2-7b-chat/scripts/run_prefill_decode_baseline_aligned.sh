#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-7}"
OUT_DIR="${ROOT}/max_speed/prefill_decode/baseline_aligned_speed"
CHECKPOINT="${ROOT}/max_speed/prefill_decode/checkpoint"
mkdir -p "${OUT_DIR}"

run() {
  local output_seq="$1" path="$2"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_phase_baseline_one.py" --checkpoint "${CHECKPOINT}" --batch 16 --input-seq 2048 --output-seq "${output_seq}" --output-json "${path}"
}

for output_seq in 1 80; do
  run "${output_seq}" "${OUT_DIR}/warmup_o${output_seq}.json"
  for index in 0 1 2 3 4; do run "${output_seq}" "${OUT_DIR}/measured_o${output_seq}_${index}.json"; done
done
"${PYTHON_BIN}" - <<PY
import json, statistics
from pathlib import Path
root = Path("${OUT_DIR}")
ttft = [json.loads((root / f"measured_o1_{i}.json").read_text())["elapsed_ms"] for i in range(5)]
e2e = [json.loads((root / f"measured_o80_{i}.json").read_text())["elapsed_ms"] for i in range(5)]
summary = {"repeats": 5, "ttft_mean_ms": statistics.mean(ttft), "ttft_median_ms": statistics.median(ttft), "e2e_mean_ms": statistics.mean(e2e), "e2e_median_ms": statistics.median(e2e), "tpot_ms": (statistics.median(e2e) - statistics.median(ttft)) / 79, "ttft_values_ms": ttft, "e2e_values_ms": e2e}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print((root / "summary.json").read_text())
PY
