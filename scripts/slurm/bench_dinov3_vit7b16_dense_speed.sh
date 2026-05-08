#!/bin/bash
#SBATCH --job-name=dinov3_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_speed_%j.out
#SBATCH --error=err/dinov3_speed_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/

PYTHONPATH=. python scripts/bench_dinov3_vit7b16_dense_speed.py \
  --batch-size 1 \
  --input-size 3 256 256 \
  --warmup 10 \
  --iters 50 \
  --output artifacts/results/dinov3_vit7b16_dense/speed.csv

