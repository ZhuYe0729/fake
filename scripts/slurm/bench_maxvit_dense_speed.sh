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

cd /data/home/scxj523/run/wja/project/my/fake/

PYTHONPATH=. python scripts/bench_maxvit_dense_speed.py \
  --batch-size 128 \
  --input-size 3 224 224 \
  --warmup 50 \
  --iters 200 \
  --dtype auto \
  --output artifacts/results/maxvit_dense/speed.csv

