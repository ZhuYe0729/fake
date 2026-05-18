#!/bin/bash
#SBATCH --job-name=prep_cutlass_runtime
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/prep_cutlass_runtime_%j.out
#SBATCH --error=err/prep_cutlass_runtime_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_nvfp4_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_nvfp4_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_bf16_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/checkpoints/dinov3_vit7b16

BACKEND="${BACKEND:-all}"
SPARSE_SOURCE_CHECKPOINT="${SPARSE_SOURCE_CHECKPOINT:-artifacts/checkpoints/dinov3_vit7b16/nvfp4_semi_structured_sparse/model.pt}"
SPARSE_BF16_SOURCE_CHECKPOINT="${SPARSE_BF16_SOURCE_CHECKPOINT:-artifacts/checkpoints/dinov3_vit7b16/semi_structured_sparse/model.pt}"
SPARSE_STORAGE_CHECKPOINT="${SPARSE_STORAGE_CHECKPOINT:-}"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

if [[ "${BACKEND}" == "all" || "${BACKEND}" == "dense_nvfp4" ]]; then
  PYTHONPATH=. python scripts/prepare_dinov3_cutlass_runtime_checkpoint.py \
    --backend dense_nvfp4 \
    --output artifacts/checkpoints/dinov3_vit7b16/cutlass_nvfp4_runtime/model.pt
fi

if [[ "${BACKEND}" == "all" || "${BACKEND}" == "sparse_nvfp4" ]]; then
  if [[ -n "${SPARSE_STORAGE_CHECKPOINT}" ]]; then
    PYTHONPATH=. python scripts/prepare_dinov3_cutlass_runtime_checkpoint.py \
      --backend sparse_nvfp4 \
      --storage-checkpoint "${SPARSE_STORAGE_CHECKPOINT}" \
      --output artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_runtime/model.pt
  else
    PYTHONPATH=. python scripts/prepare_dinov3_cutlass_runtime_checkpoint.py \
      --backend sparse_nvfp4 \
      --source-checkpoint "${SPARSE_SOURCE_CHECKPOINT}" \
      --no-prune \
      --output artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_runtime/model.pt
  fi
fi

if [[ "${BACKEND}" == "all" || "${BACKEND}" == "sparse_bf16" ]]; then
  PYTHONPATH=. python scripts/prepare_dinov3_cutlass_runtime_checkpoint.py \
    --backend sparse_bf16 \
    --source-checkpoint "${SPARSE_BF16_SOURCE_CHECKPOINT}" \
    --no-prune \
    --output artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_bf16_runtime/model.pt
fi
