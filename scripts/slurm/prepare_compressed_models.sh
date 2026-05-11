#!/bin/bash
#SBATCH --job-name=prepare_compress
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/prepare_compress_%j.out
#SBATCH --error=err/prepare_compress_%j.err

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
MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
METHODS="${METHODS:-nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse}"
if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
  DEFAULT_CALIB_BATCH_SIZE=4
else
  DEFAULT_CALIB_BATCH_SIZE=16
fi
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-${DEFAULT_CALIB_BATCH_SIZE}}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
which nvidia-smi || true
nvidia-smi || true

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"


for METHOD in ${METHODS}; do
  echo "Preparing ${MODEL} ${MAXVIT_VARIANT} ${METHOD}"
  if [[ "${MODEL}" == "maxvit" ]]; then
    PYTHONPATH=. python scripts/prepare_compressed_model.py \
      --model "${MODEL}" \
      --maxvit-variant "${MAXVIT_VARIANT}" \
      --calib-batch-size "${CALIB_BATCH_SIZE}" \
      --method "${METHOD}"
  else
    PYTHONPATH=. python scripts/prepare_compressed_model.py \
      --model "${MODEL}" \
      --method "${METHOD}"
  fi
done
