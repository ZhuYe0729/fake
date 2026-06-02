#!/bin/bash
# Comprehensive Qwen3.5-0.8B benchmark sweep
# Configs: batch_size × input_tokens × output_tokens
# Tests: speed, breakdown-coarse, breakdown-fine
set -euo pipefail

MODEL_NAME="Qwen3.5-0.8B"
OUTDIR="artifacts/results/benchmarks/module/${MODEL_NAME}"
mkdir -p "$OUTDIR"

SCRIPT="scripts/bench_qwen3_5_speed.py"

BATCH_SIZES="1 2 4 8 32 64"
INPUT_TOKENS="128 512 4096 8192 16384"
OUTPUT_TOKENS="32 512"
WARMUP=3
ITERS=10

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

# ── Speed test (no hooks) ───────────────────────────────────────────
log "=== Phase 1/3: Speed benchmark (no hooks) ==="
python "$SCRIPT" \
    --batch-sizes $BATCH_SIZES \
    --input-tokens $INPUT_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --warmup $WARMUP --iters $ITERS \
    --output-csv "${OUTDIR}/speed.csv" \
    --verbose
log "Phase 1 done → ${OUTDIR}/speed.csv"

# ── Coarse breakdown ────────────────────────────────────────────────
log "=== Phase 2/3: Coarse breakdown ==="
python "$SCRIPT" \
    --batch-sizes $BATCH_SIZES \
    --input-tokens $INPUT_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --warmup $WARMUP --iters $ITERS \
    --breakdown --breakdown-mode coarse \
    --output-csv "${OUTDIR}/breakdown_coarse.csv" \
    --verbose
log "Phase 2 done → ${OUTDIR}/breakdown_coarse.csv"

# ── Fine breakdown ──────────────────────────────────────────────────
log "=== Phase 3/3: Fine breakdown ==="
python "$SCRIPT" \
    --batch-sizes $BATCH_SIZES \
    --input-tokens $INPUT_TOKENS \
    --output-tokens $OUTPUT_TOKENS \
    --warmup $WARMUP --iters $ITERS \
    --breakdown --breakdown-mode fine \
    --output-csv "${OUTDIR}/breakdown_fine.csv" \
    --verbose
log "Phase 3 done → ${OUTDIR}/breakdown_fine.csv"

log "=== All done! Results in ${OUTDIR}/ ==="
ls -lh "${OUTDIR}/"
