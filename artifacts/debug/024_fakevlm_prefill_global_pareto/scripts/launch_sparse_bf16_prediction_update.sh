#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/024_fakevlm_prefill_global_pareto}"
CONDA_BIN="${CONDA_BIN:-/home/agent/wja/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-cospaq}"
LOG_DIR="$ROOT/logs/sparse_bf16_prediction_update"
OUTPUT_DIR="$ROOT/prediction_vs_actual/corrected_nll_batch16_refined_sparse_bf16"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "[prediction-update] launching P19-P21 corrected NLL on GPUs 0-2"
pids=()
for point in 19 20 21; do
  gpu="$((point - 19))"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_loss.py" \
    --output-root "$ROOT" \
    --policies validation \
    --points "16:$point" \
    --calib-samples 128 \
    --batch-size 2 \
    --calib-batch-size 1 \
    --workers 1 \
    --gpu 0 \
    >> "$LOG_DIR/loss_point_${point}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "[prediction-update] at least one NLL job failed" >&2
  exit 1
fi

echo "[prediction-update] NLL complete; building 11-policy comparison"
MPLCONFIGDIR=/tmp/matplotlib-024-prediction-refined "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$ROOT/scripts/build_prediction_comparison.py" \
  --output-root "$ROOT" \
  --batches 16 \
  --output-dir "$OUTPUT_DIR"
echo "[prediction-update] complete"
