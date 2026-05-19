#!/bin/bash
#SBATCH --job-name=maxvit_4over6_prep
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/maxvit_4over6_prep_%j.out
#SBATCH --error=err/maxvit_4over6_prep_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/checkpoints

MAXVIT_VARIANTS="${MAXVIT_VARIANTS:-tiny small base large}"
METHODS="${METHODS:-nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-}"
CALIB_SEEDS="${CALIB_SEEDS:-}"
CALIB_SHUFFLE="${CALIB_SHUFFLE:-0}"

read -r -a VARIANT_ARGS <<< "${MAXVIT_VARIANTS}"
read -r -a METHOD_ARGS <<< "${METHODS}"
if [[ -n "${CALIB_SEEDS}" ]]; then
  read -r -a SEED_ARGS <<< "${CALIB_SEEDS}"
else
  SEED_ARGS=("")
fi

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

for variant in "${VARIANT_ARGS[@]}"; do
  if [[ -n "${CALIB_BATCH_SIZE}" ]]; then
    batch_size="${CALIB_BATCH_SIZE}"
  elif [[ "${variant}" == "large" ]]; then
    batch_size=4
  else
    batch_size=16
  fi

  for method in "${METHOD_ARGS[@]}"; do
    for seed in "${SEED_ARGS[@]}"; do
      EXTRA_ARGS=()
      if [[ -n "${seed}" ]]; then
        EXTRA_ARGS+=(--seed "${seed}" --output-dir "artifacts/checkpoints/maxvit_${variant}/${method}_seed${seed}")
      fi
      if [[ "${CALIB_SHUFFLE}" == "1" ]]; then
        EXTRA_ARGS+=(--calib-shuffle)
      fi

      echo "Preparing MaxViT ${variant} ${method} with calib_batch_size=${batch_size} seed=${seed:-default} calib_shuffle=${CALIB_SHUFFLE}"
      PYTHONPATH=. python scripts/prepare_compressed_model.py \
        --model maxvit \
        --maxvit-variant "${variant}" \
        --calib-batch-size "${batch_size}" \
        --method "${method}" \
        "${EXTRA_ARGS[@]}"
    done
  done
done
