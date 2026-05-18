#!/bin/bash
#SBATCH --job-name=maxvit_4over6_acc
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/maxvit_4over6_acc_%j.out
#SBATCH --error=err/maxvit_4over6_acc_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/results

MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
METHOD="${METHOD:-nvfp4_4over6_unstructured_sparse}"
CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/maxvit_${MAXVIT_VARIANT}/${METHOD}/model.pt}"
ACTIVATION_SCALE_RULE="${ACTIVATION_SCALE_RULE:-four_over_six_mse}"
NO_ACTIVATION_QUANT="${NO_ACTIVATION_QUANT:-0}"

if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
  DEFAULT_BATCH_SIZE=16
else
  DEFAULT_BATCH_SIZE=128
fi
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"

ARGS=(
  --variant "${MAXVIT_VARIANT}" \
  --batch-size "${BATCH_SIZE}" \
  --checkpoint "${CHECKPOINT}" \
  --method "${METHOD}" \
  --activation-scale-rule "${ACTIVATION_SCALE_RULE}" \
  --output "artifacts/results/maxvit_${MAXVIT_VARIANT}_4over6/accuracy.csv"
)
if [[ "${NO_ACTIVATION_QUANT}" != "1" ]]; then
  ARGS+=(--activation-quant)
fi

PYTHONPATH=. python scripts/eval_maxvit_dense_accuracy.py "${ARGS[@]}"
