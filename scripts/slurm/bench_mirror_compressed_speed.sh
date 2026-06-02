#!/bin/bash
#SBATCH --job-name=mirror_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/mirror_speed_%j.out
#SBATCH --error=err/mirror_speed_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_marlin_nvfp4_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/

METHODS="${METHODS:-${METHOD:-dense nvfp4 marlin_nvfp4 semi_structured_sparse nvfp4_semi_structured_sparse}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
OUTPUT="${OUTPUT:-artifacts/results/mirror_compressed/speed.csv}"

for method in ${METHODS}; do
  CHECKPOINT_ARG=()
  if [[ "${method}" != "dense" ]]; then
    checkpoint="${CHECKPOINT:-artifacts/checkpoints/mirror/${method}/model.pt}"
    CHECKPOINT_ARG=(--checkpoint "${checkpoint}")
  fi
  echo "[mirror speed] method=${method}"
  PYTHONPATH=. python scripts/bench_mirror_compressed_speed.py \
    --method "${method}" \
    "${CHECKPOINT_ARG[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --output "${OUTPUT}"
done
