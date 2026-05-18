#!/bin/bash
#SBATCH --job-name=dinov3_4over6_spd
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_4over6_spd_%j.out
#SBATCH --error=err/dinov3_4over6_spd_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/

METHOD="${METHOD:-nvfp4_4over6_unstructured_sparse}"
CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/dinov3_vit7b16/${METHOD}/model.pt}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
ACTIVATION_SCALE_RULE="${ACTIVATION_SCALE_RULE:-four_over_six_mse}"
NO_ACTIVATION_QUANT="${NO_ACTIVATION_QUANT:-0}"

mkdir -p out err artifacts/results/dinov3_vit7b16_4over6_unstructured_sparse artifacts/results/dinov3_vit7b16_4over6_semi_structured_sparse

ARGS=(
  --method "${METHOD}"
  --checkpoint "${CHECKPOINT}"
  --batch-size "${BATCH_SIZE}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --activation-scale-rule "${ACTIVATION_SCALE_RULE}"
)
if [[ "${NO_ACTIVATION_QUANT}" == "1" ]]; then
  ARGS+=(--no-activation-quant)
fi

PYTHONPATH=. python scripts/bench_dinov3_vit7b16_four_over_six_speed.py "${ARGS[@]}"
