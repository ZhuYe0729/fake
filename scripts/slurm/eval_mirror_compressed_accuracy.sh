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

METHODS="${METHODS:-${METHOD:-dense}}"
BENCHMARKS="${BENCHMARKS:-Chameleon GenImage}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT="${OUTPUT:-artifacts/results/mirror_compressed/accuracy.csv}"
LIMIT_ARG=()
if [[ -n "${LIMIT_PER_CLASS:-}" ]]; then
  LIMIT_ARG=(--limit-per-class "${LIMIT_PER_CLASS}")
fi
GENIMAGE_ARG=()
if [[ "${PREFER_EXTRACTED_GENIMAGE:-1}" == "1" ]]; then
  GENIMAGE_ARG=(--prefer-extracted-genimage)
fi

for method in ${METHODS}; do
  CHECKPOINT_ARG=()
  if [[ "${method}" != "dense" ]]; then
    checkpoint="${CHECKPOINT:-artifacts/checkpoints/mirror/${method}/model.pt}"
    CHECKPOINT_ARG=(--checkpoint "${checkpoint}")
  fi
  echo "[mirror accuracy] method=${method}"
  PYTHONPATH=. python scripts/eval_mirror_compressed_accuracy.py \
    --method "${method}" \
    "${CHECKPOINT_ARG[@]}" \
    --benchmarks ${BENCHMARKS} \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --output "${OUTPUT}" \
    "${GENIMAGE_ARG[@]}" \
    "${LIMIT_ARG[@]}"
done
