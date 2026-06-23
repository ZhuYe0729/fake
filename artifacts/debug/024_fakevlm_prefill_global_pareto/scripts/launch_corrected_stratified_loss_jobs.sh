#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/024_fakevlm_prefill_global_pareto}"
CONDA_BIN="${CONDA_BIN:-/home/agent/wja/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-cospaq}"
LOG_DIR="$ROOT/logs/corrected_quality_model"
ARCHIVE_DIR="$ROOT/archive_invalid_nll_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR/quality" "$ROOT/quality"

archive_path() {
  local path="$1"
  local destination="$2"
  if [[ -e "$path" ]]; then
    mv "$path" "$destination"
  fi
}

archive_path "$ROOT/quality/stratified_loss.csv" "$ARCHIVE_DIR/quality/stratified_loss.csv"
archive_path "$ROOT/quality/stratified_loss.csv.lock" "$ARCHIVE_DIR/quality/stratified_loss.csv.lock"
archive_path "$ROOT/quality/stratified_loss_metadata.json" "$ARCHIVE_DIR/quality/stratified_loss_metadata.json"
archive_path "$ROOT/quality/stratified_loss_points" "$ARCHIVE_DIR/quality/stratified_loss_points"
archive_path "$ROOT/global_coefficients" "$ARCHIVE_DIR/global_coefficients"

echo "[launcher] archived invalid NLL artifacts under $ARCHIVE_DIR"
echo "[launcher] measuring corrected dense NLL baseline on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_loss.py" \
  --output-root "$ROOT" \
  --policies stratified \
  --policy-indices "0" \
  --calib-samples 128 \
  --batch-size 4 \
  --calib-batch-size 1 \
  --workers 1 \
  --gpu 0 \
  >> "$LOG_DIR/stratified_loss_dense_gpu0.log" 2>&1

echo "[launcher] dense baseline finished; launching corrected stratified loss shards"
GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
SHARDS=(
  "1,2,3,4,5,6,7,8"
  "9,10,11,12,13,14,15,16"
  "17,18,19,20,21,22,23,24"
  "25,26,27,28,29,30,31,32"
  "33,34,35,36,37,38,39,40"
  "41,42,43,44,45,46,47,48"
  "49,50,51,52,53,54,55,56"
  "57,58,59,60"
)

for i in "${!SHARDS[@]}"; do
  gpu="${GPU_LIST[$i]}"
  indices="${SHARDS[$i]}"
  log="$LOG_DIR/stratified_loss_shard_${i}_gpu${gpu}.log"
  echo "[launcher] shard=$i gpu=$gpu indices=$indices log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_loss.py" \
    --output-root "$ROOT" \
    --policies stratified \
    --policy-indices "$indices" \
    --calib-samples 128 \
    --batch-size 4 \
    --calib-batch-size 1 \
    --workers 1 \
    --gpu 0 \
    >> "$log" 2>&1 &
done

wait
echo "[launcher] corrected stratified NLL finished; fitting quality model"
"$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/fit_quality_model.py" \
  --output-root "$ROOT" \
  >> "$LOG_DIR/fit_quality_model.log" 2>&1
echo "[launcher] corrected quality-model stage finished"
