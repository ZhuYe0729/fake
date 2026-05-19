#!/bin/bash
#SBATCH --job-name=dinov3_seed_acc
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_seed_acc_%j.out
#SBATCH --error=err/dinov3_seed_acc_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/results/dinov3_vit7b16_compressed artifacts/results/dinov3_vit7b16_4over6_unstructured_sparse artifacts/results/dinov3_vit7b16_4over6_semi_structured_sparse

METHODS="${METHODS:-nvfp4_unstructured_sparse nvfp4_semi_structured_sparse nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse}"
CALIB_SEEDS="${CALIB_SEEDS:-1 2 3}"
CALIB_SHUFFLE="${CALIB_SHUFFLE:-1}"
CALIB_SAMPLES="${CALIB_SAMPLES:-16}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACTIVATION_MODE="${ACTIVATION_MODE:-auto}"

read -r -a METHOD_ARGS <<< "${METHODS}"
read -r -a SEED_ARGS <<< "${CALIB_SEEDS}"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

for method in "${METHOD_ARGS[@]}"; do
  for seed in "${SEED_ARGS[@]}"; do
    ARGS=(
      --method "${method}"
      --seed "${seed}"
      --calib-samples "${CALIB_SAMPLES}"
      --calib-batch-size "${CALIB_BATCH_SIZE}"
      --eval-batch-size "${EVAL_BATCH_SIZE}"
      --num-workers "${NUM_WORKERS}"
      --activation-mode "${ACTIVATION_MODE}"
    )
    if [[ "${CALIB_SHUFFLE}" == "1" ]]; then
      ARGS+=(--calib-shuffle)
    fi
    echo "Evaluating DINOv3 ${method} seed=${seed} calib_shuffle=${CALIB_SHUFFLE} activation_mode=${ACTIVATION_MODE}"
    PYTHONPATH=. python scripts/eval_dinov3_vit7b16_seeded_compression_accuracy.py "${ARGS[@]}"
  done
done
