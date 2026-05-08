#!/bin/bash
#SBATCH --job-name=eval_compress
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/eval_compress_%j.out
#SBATCH --error=err/eval_compress_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/

MODEL="${MODEL:-maxvit}"
METHOD="${METHOD:-nvfp4}"
CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/${MODEL}/${METHOD}/model.pt}"

if [[ "${MODEL}" == "maxvit" ]]; then
  PYTHONPATH=. python scripts/eval_maxvit_dense_accuracy.py \
    --checkpoint "${CHECKPOINT}" \
    --method "${METHOD}" \
    --output "artifacts/results/maxvit_compressed/accuracy.csv"
elif [[ "${MODEL}" == "dinov3_vit7b16" ]]; then
  PYTHONPATH=. python scripts/eval_dinov3_vit7b16_dense_accuracy.py \
    --checkpoint "${CHECKPOINT}" \
    --method "${METHOD}" \
    --output "artifacts/results/dinov3_vit7b16_compressed/accuracy.csv"
else
  echo "Unsupported MODEL=${MODEL}" >&2
  exit 1
fi
