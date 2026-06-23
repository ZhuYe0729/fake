#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/024_fakevlm_prefill_global_pareto}"
CONDA_BIN="${CONDA_BIN:-/home/agent/wja/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-cospaq}"
POINTS="19,20,21"
LOG_DIR="$ROOT/logs/sparse_bf16_frontier_refinement"

mkdir -p "$LOG_DIR"

echo "[refinement] measuring P19-P21 speed serially on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$ROOT/scripts/validate_pareto_speed.py" \
  --output-root "$ROOT" \
  --batch-size 16 \
  --points "$POINTS" \
  --calib-samples 128 \
  --workers 1 \
  --gpu 0 \
  >> "$LOG_DIR/speed_points_19_20_21_gpu0.log" 2>&1

echo "[refinement] speed complete; measuring full FakeClue accuracy on GPUs 0-2"
for point in 19 20 21; do
  gpu="$((point - 19))"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_quality.py" \
    --output-root "$ROOT" \
    --policies validation \
    --points "16:$point" \
    --calib-samples 128 \
    --batch-size 8 \
    --calib-batch-size 1 \
    --workers 1 \
    --gpu 0 \
    >> "$LOG_DIR/quality_point_${point}_gpu${gpu}.log" 2>&1 &
done
wait

echo "[refinement] validation complete; rebuilding joined results and refined report"
MPLCONFIGDIR=/tmp/matplotlib-024-refined-sparse-bf16 "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$ROOT/scripts/summarize_validation.py" \
  --output-root "$ROOT"
MPLCONFIGDIR=/tmp/matplotlib-024-refined-sparse-bf16 "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$ROOT/scripts/build_report_plots.py" \
  --output-root "$ROOT" \
  --filename-suffix refined_sparse_bf16

echo "[refinement] complete"
