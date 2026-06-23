#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/debug/024_fakevlm_prefill_global_pareto}"
CONDA_BIN="${CONDA_BIN:-/home/agent/wja/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-cospaq}"
BATCH=16
LOG_DIR="$ROOT/logs/corrected_batch16_validation"
ARCHIVE_DIR="$ROOT/archive_invalid_nll_20260621_142347"
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR/validation" "$ARCHIVE_DIR/quality" "$ARCHIVE_DIR/summary" "$ROOT/validation" "$ROOT/quality"

archive_path() {
  local path="$1"
  local destination="$2"
  if [[ -e "$path" ]]; then
    mv "$path" "$destination"
  fi
}

archive_path "$ROOT/validation/pareto_speed_validation.csv" "$ARCHIVE_DIR/validation/pareto_speed_validation.csv"
archive_path "$ROOT/validation/pareto_speed_validation.csv.lock" "$ARCHIVE_DIR/validation/pareto_speed_validation.csv.lock"
archive_path "$ROOT/validation/pareto_speed_validation_batch_16_metadata.json" "$ARCHIVE_DIR/validation/pareto_speed_validation_batch_16_metadata.json"
archive_path "$ROOT/validation/pareto_validation_joined.csv" "$ARCHIVE_DIR/validation/pareto_validation_joined.csv"
archive_path "$ROOT/quality/validation_quality.csv" "$ARCHIVE_DIR/quality/validation_quality.csv"
archive_path "$ROOT/quality/validation_quality.csv.lock" "$ARCHIVE_DIR/quality/validation_quality.csv.lock"
archive_path "$ROOT/quality/validation_quality_metadata.json" "$ARCHIVE_DIR/quality/validation_quality_metadata.json"
archive_path "$ROOT/quality/validation" "$ARCHIVE_DIR/quality/validation"
archive_path "$ROOT/quality/validation_loss.csv" "$ARCHIVE_DIR/quality/validation_loss.csv"
archive_path "$ROOT/quality/validation_loss.csv.lock" "$ARCHIVE_DIR/quality/validation_loss.csv.lock"
archive_path "$ROOT/quality/validation_loss_metadata.json" "$ARCHIVE_DIR/quality/validation_loss_metadata.json"
archive_path "$ROOT/quality/validation_loss_points" "$ARCHIVE_DIR/quality/validation_loss_points"
archive_path "$ROOT/summary/analysis.md" "$ARCHIVE_DIR/summary/analysis.md"

mapfile -t POINTS < <(python -c "import csv; print(*[int(float(r['point_index'])) for r in csv.DictReader(open('$ROOT/validation/selected_pareto_points.csv')) if int(float(r['batch_size'])) == $BATCH], sep='\\n')")
if [[ "${#POINTS[@]}" -ne 8 ]] || [[ "${POINTS[0]}" -ne 0 ]]; then
  echo "expected 8 batch-16 points starting with dense point 0; got: ${POINTS[*]}" >&2
  exit 1
fi

echo "[launcher] measuring batch-16 speed on GPU 0; no concurrent GPU tasks"
CUDA_VISIBLE_DEVICES=0 "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_pareto_speed.py" \
  --output-root "$ROOT" \
  --batch-size "$BATCH" \
  --points validation \
  --calib-samples 128 \
  --workers 1 \
  --gpu 0 \
  >> "$LOG_DIR/speed_batch16_gpu0.log" 2>&1

echo "[launcher] speed finished; measuring corrected dense NLL on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_loss.py" \
  --output-root "$ROOT" \
  --policies validation \
  --points "16:0" \
  --calib-samples 128 \
  --batch-size 2 \
  --calib-batch-size 1 \
  --workers 1 \
  --gpu 0 \
  >> "$LOG_DIR/loss_dense_gpu0.log" 2>&1

echo "[launcher] dense NLL finished; launching seven non-dense NLL points"
for i in "${!POINTS[@]}"; do
  point="${POINTS[$i]}"
  if [[ "$point" -eq 0 ]]; then
    continue
  fi
  gpu="$((i - 1))"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_loss.py" \
    --output-root "$ROOT" \
    --policies validation \
    --points "16:$point" \
    --calib-samples 128 \
    --batch-size 2 \
    --calib-batch-size 1 \
    --workers 1 \
    --gpu 0 \
    >> "$LOG_DIR/loss_point_${point}_gpu${gpu}.log" 2>&1 &
done
wait

echo "[launcher] NLL finished; launching eight FakeClue accuracy points"
for i in "${!POINTS[@]}"; do
  point="${POINTS[$i]}"
  gpu="$i"
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/validate_policy_quality.py" \
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

echo "[launcher] validation finished; building batch-16 summary and report"
MPLCONFIGDIR=/tmp/matplotlib-024-corrected-b16 "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/summarize_validation.py" \
  --output-root "$ROOT"
MPLCONFIGDIR=/tmp/matplotlib-024-corrected-b16 "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT/scripts/build_report_plots.py" \
  --output-root "$ROOT" \
  --filename-suffix corrected_nll_batch16
echo "[launcher] corrected batch-16 validation complete"
