#!/bin/bash
#SBATCH --job-name=dinov3_4over6_prep
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_4over6_prep_%j.out
#SBATCH --error=err/dinov3_4over6_prep_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/

mkdir -p out err artifacts/checkpoints/dinov3_vit7b16

METHODS="${METHODS:-nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

PYTHONPATH=. python scripts/prepare_dinov3_vit7b16_four_over_six_checkpoints.py \
  --methods ${METHODS} \
  --calib-batch-size "${CALIB_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
