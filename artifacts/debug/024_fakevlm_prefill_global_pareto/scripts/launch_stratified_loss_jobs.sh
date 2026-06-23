#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/024_fakevlm_prefill_global_pareto}"
CONDA_BIN="${CONDA_BIN:-/home/agent/wja/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-cospaq}"
LOG_DIR="$ROOT/logs/loss"
mkdir -p "$LOG_DIR"

LOSS_CSV="$ROOT/quality/stratified_loss.csv"
echo "[launcher] waiting for dense baseline in $LOSS_CSV"
while true; do
  if [[ -s "$LOSS_CSV" ]] && grep -q '^policy_000,' "$LOSS_CSV"; then
    break
  fi
  sleep 30
done
echo "[launcher] dense baseline found; launching stratified loss shards"

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
  log="$LOG_DIR/stratified_loss_gpu${gpu}.log"
  echo "[launcher] gpu=$gpu indices=$indices log=$log"
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
echo "[launcher] all stratified loss shards finished"
