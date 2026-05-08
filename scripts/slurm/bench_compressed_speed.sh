#!/bin/bash
#SBATCH --job-name=bench_compress
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/bench_compress_%j.out
#SBATCH --error=err/bench_compress_%j.err

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
  PYTHONPATH=. python scripts/bench_maxvit_dense_speed.py \
    --checkpoint "${CHECKPOINT}" \
    --method "${METHOD}" \
    --output "artifacts/results/maxvit_compressed/speed.csv"
elif [[ "${MODEL}" == "dinov3_vit7b16" ]]; then
  PYTHONPATH=. python scripts/bench_dinov3_vit7b16_dense_speed.py \
    --checkpoint "${CHECKPOINT}" \
    --method "${METHOD}" \
    --output "artifacts/results/dinov3_vit7b16_compressed/speed.csv"
else
  echo "Unsupported MODEL=${MODEL}" >&2
  exit 1
fi
