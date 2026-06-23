#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/024_fakevlm_prefill_global_pareto}"
CONDA_BIN="${CONDA_BIN:-/home/agent/wja/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-cospaq}"
LOG_DIR="$ROOT/logs/validation"
ARCHIVE_DIR="$ROOT/validation/archive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR" "$ROOT/validation" "$ROOT/quality"

archive_file() {
  local path="$1"
  if [[ -e "$path" ]]; then
    mv "$path" "$ARCHIVE_DIR/$(basename "$path")"
  fi
}

archive_file "$ROOT/validation/pareto_speed_validation.csv"
archive_file "$ROOT/validation/pareto_speed_validation.csv.lock"
archive_file "$ROOT/quality/validation_quality.csv"
archive_file "$ROOT/quality/validation_quality.csv.lock"

echo "[launcher] archived stale validation CSVs under $ARCHIVE_DIR"
echo "[launcher] launching speed validation; one batch per GPU"

SPEED_BATCHES=(${SPEED_BATCHES:-1 2 4 8 16})
SPEED_GPUS=(${SPEED_GPUS:-0 1 2 3 4})

for i in "${!SPEED_BATCHES[@]}"; do
  batch="${SPEED_BATCHES[$i]}"
  gpu="${SPEED_GPUS[$i]}"
  log="$LOG_DIR/speed_batch_${batch}_gpu${gpu}.log"
  echo "[launcher] speed batch=$batch gpu=$gpu log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_pareto_speed.py" \
    --output-root "$ROOT" \
    --batch-size "$batch" \
    --points validation \
    --calib-samples 128 \
    --workers 1 \
    --gpu 0 \
    >> "$log" 2>&1 &
done

wait
echo "[launcher] speed validation finished; launching FakeClue accuracy shards"

QUALITY_GPUS=(${QUALITY_GPUS:-0 1 2 3 4 5 6 7})
QUALITY_SHARDS=(
  "1:0,1:5,1:9,1:13,1:18"
  "1:22,1:26,1:30,2:0,2:5"
  "2:9,2:13,2:18,2:22,2:26"
  "2:30,4:0,4:5,4:9,4:13"
  "4:18,4:22,4:26,4:30,8:0"
  "8:5,8:9,8:13,8:18,8:22"
  "8:26,8:30,16:0,16:5,16:9"
  "16:13,16:18,16:22,16:26,16:30"
)

for i in "${!QUALITY_SHARDS[@]}"; do
  gpu="${QUALITY_GPUS[$i]}"
  points="${QUALITY_SHARDS[$i]}"
  log="$LOG_DIR/quality_shard_${i}_gpu${gpu}.log"
  echo "[launcher] quality shard=$i gpu=$gpu points=$points log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_quality.py" \
    --output-root "$ROOT" \
    --policies validation \
    --points "$points" \
    --calib-samples 128 \
    --batch-size 8 \
    --calib-batch-size 1 \
    --workers 1 \
    --gpu 0 \
    >> "$log" 2>&1 &
done

wait
echo "[launcher] selected policy validation finished"
