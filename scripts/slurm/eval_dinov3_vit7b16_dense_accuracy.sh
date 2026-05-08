#!/bin/bash
#SBATCH --job-name=dinov3_acc
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_acc_%j.out
#SBATCH --error=err/dinov3_acc_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/

PYTHONPATH=. python scripts/eval_dinov3_vit7b16_dense_accuracy.py \
  --batch-size 1 \
  --num-workers 4 \
  --resize-size 256 \
  --output artifacts/results/dinov3_vit7b16_dense/accuracy.csv

