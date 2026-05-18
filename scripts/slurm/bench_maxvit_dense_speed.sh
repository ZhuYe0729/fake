#!/bin/bash
#SBATCH --job-name=maxvit_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/maxvit_speed_%j.out
#SBATCH --error=err/maxvit_speed_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err

MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
  DEFAULT_BATCH_SIZE=16
else
  DEFAULT_BATCH_SIZE=128
fi
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-200}"
INPUT_SIZE="${INPUT_SIZE:-}"
DTYPE="${DTYPE:-auto}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT="${OUTPUT:-artifacts/results/maxvit_${MAXVIT_VARIANT}_dense/speed.csv}"

EXTRA_ARGS=()
if [[ -n "${INPUT_SIZE}" ]]; then
  read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
  EXTRA_ARGS+=(--input-size "${INPUT_SIZE_ARGS[@]}")
fi
if [[ -n "${MODEL_PATH}" ]]; then
  EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
fi
mkdir -p "$(dirname "${OUTPUT}")"

PYTHONPATH=. python scripts/bench_maxvit_dense_speed.py \
  --variant "${MAXVIT_VARIANT}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --dtype "${DTYPE}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
