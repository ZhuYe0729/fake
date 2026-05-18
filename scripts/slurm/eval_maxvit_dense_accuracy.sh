#!/bin/bash
#SBATCH --job-name=maxvit_acc
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/maxvit_acc_%j.out
#SBATCH --error=err/maxvit_acc_%j.err

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
NUM_WORKERS="${NUM_WORKERS:-8}"
DTYPE="${DTYPE:-auto}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT="${OUTPUT:-artifacts/results/maxvit_${MAXVIT_VARIANT}_dense/accuracy.csv}"

EXTRA_ARGS=()
if [[ -n "${MODEL_PATH}" ]]; then
  EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
fi
mkdir -p "$(dirname "${OUTPUT}")"

PYTHONPATH=. python scripts/eval_maxvit_dense_accuracy.py \
  --variant "${MAXVIT_VARIANT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --dtype "${DTYPE}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
