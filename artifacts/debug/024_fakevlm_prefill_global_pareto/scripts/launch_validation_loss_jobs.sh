#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/024_fakevlm_prefill_global_pareto}"
CONDA_BIN="${CONDA_BIN:-/home/agent/wja/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-cospaq}"
LOG_DIR="$ROOT/logs/prediction_vs_actual"
ARCHIVE_DIR="$ROOT/quality/archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR" "$ROOT/quality"

for path in "$ROOT/quality/validation_loss.csv" "$ROOT/quality/validation_loss.csv.lock"; do
  if [[ -e "$path" ]]; then
    mv "$path" "$ARCHIVE_DIR/$(basename "$path")"
  fi
done

echo "[launcher] measuring validation dense NLL baseline on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_loss.py" \
  --output-root "$ROOT" \
  --policies validation \
  --points "1:0" \
  --calib-samples 128 \
  --batch-size 2 \
  --calib-batch-size 1 \
  --workers 1 \
  --gpu 0 \
  >> "$LOG_DIR/validation_loss_dense_gpu0.log" 2>&1

echo "[launcher] dense NLL baseline finished; launching remaining validation loss shards"
QUALITY_GPUS=(${QUALITY_GPUS:-0 1 2 3 4 5 6 7})
QUALITY_SHARDS=(
  "1:5,1:9,1:13,1:18,1:22"
  "1:26,1:30,2:0,2:5,2:9"
  "2:13,2:18,2:22,2:26,2:30"
  "4:0,4:5,4:9,4:13,4:18"
  "4:22,4:26,4:30,8:0,8:5"
  "8:9,8:13,8:18,8:22,8:26"
  "8:30,16:0,16:5,16:9,16:13"
  "16:18,16:22,16:26,16:30"
)

for i in "${!QUALITY_SHARDS[@]}"; do
  gpu="${QUALITY_GPUS[$i]}"
  points="${QUALITY_SHARDS[$i]}"
  log="$LOG_DIR/validation_loss_shard_${i}_gpu${gpu}.log"
  echo "[launcher] loss shard=$i gpu=$gpu points=$points log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_loss.py" \
    --output-root "$ROOT" \
    --policies validation \
    --points "$points" \
    --calib-samples 128 \
    --batch-size 2 \
    --calib-batch-size 1 \
    --workers 1 \
    --gpu 0 \
    >> "$log" 2>&1 &
done

wait
echo "[launcher] validation loss finished"

echo "[launcher] building prediction-versus-actual artifacts"
MPLCONFIGDIR=/tmp/matplotlib-024-prediction-vs-actual "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/build_prediction_comparison.py" \
  --output-root "$ROOT"
echo "[launcher] prediction-versus-actual artifacts finished"
