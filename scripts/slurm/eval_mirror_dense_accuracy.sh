#!/bin/bash
#SBATCH --job-name=mirror_acc
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/mirror_acc_%j.out
#SBATCH --error=err/mirror_acc_%j.err

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

BENCHMARKS="${BENCHMARKS:-Chameleon GenImage}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT="${OUTPUT:-artifacts/results/mirror_dense/accuracy.csv}"
CHAMELEON_ROOT="${CHAMELEON_ROOT:-/data/home/scxj523/run/wja/data/datasets/Chameleon/test}"
GENIMAGE_ROOT="${GENIMAGE_ROOT:-/data/home/scxj523/run/wja/data/datasets/genimage-validation}"
GENIMAGE_ZIP="${GENIMAGE_ZIP:-/data/home/scxj523/run/wja/data/datasets/genimage-validation/genimage-validation.zip}"
MODEL_PATH="${MODEL_PATH:-/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight/checkpoint-h-cur.pth}"
MEMORY_PATH="${MEMORY_PATH:-/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight/mirror_phase1.pth}"
BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight/dinov3-huge}"
USE_AMP="${USE_AMP:-1}"
LIMIT_PER_CLASS="${LIMIT_PER_CLASS:-}"
PREFER_EXTRACTED_GENIMAGE="${PREFER_EXTRACTED_GENIMAGE:-0}"

EXTRA_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  EXTRA_ARGS+=(--use-amp)
fi
if [[ -n "${LIMIT_PER_CLASS}" ]]; then
  EXTRA_ARGS+=(--limit-per-class "${LIMIT_PER_CLASS}")
fi
if [[ "${PREFER_EXTRACTED_GENIMAGE}" == "1" ]]; then
  EXTRA_ARGS+=(--prefer-extracted-genimage)
fi

mkdir -p "$(dirname "${OUTPUT}")"

read -r -a BENCHMARK_ARRAY <<< "${BENCHMARKS}"

PYTHONPATH=. python -u scripts/eval_mirror_dense_accuracy.py \
  --benchmarks "${BENCHMARK_ARRAY[@]}" \
  --chameleon-root "${CHAMELEON_ROOT}" \
  --genimage-root "${GENIMAGE_ROOT}" \
  --genimage-zip "${GENIMAGE_ZIP}" \
  --model-path "${MODEL_PATH}" \
  --memory-path "${MEMORY_PATH}" \
  --backbone-path "${BACKBONE_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
