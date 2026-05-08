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

cd /data/home/scxj523/run/wja/project/my/fake/

PYTHONPATH=. python scripts/eval_maxvit_dense_accuracy.py \
  --batch-size 128 \
  --num-workers 8 \
  --dtype auto \
  --output artifacts/results/maxvit_dense/accuracy.csv

