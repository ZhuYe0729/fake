#!/bin/bash
#SBATCH --job-name=dinov3_sparse_nvfp4_acc
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_sparse_nvfp4_acc_%j.out
#SBATCH --error=err/dinov3_sparse_nvfp4_acc_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_nvfp4_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESIZE_SIZE="${RESIZE_SIZE:-256}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
OUTPUT="${OUTPUT:-artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/accuracy.csv}"
BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m}"
HEAD_PATH="${HEAD_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3_vit7b16_imagenet1k_linear_head/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth}"
DATASET_ROOT="${DATASET_ROOT:-/data/home/scxj523/run/wja/data/datasets/imagenet_val}"
CSV="${CSV:-val.csv}"
ZIP="${ZIP:-imagenet_val.zip}"
CHECKPOINT="${CHECKPOINT:-}"
NO_PRUNE="${NO_PRUNE:-0}"
RUNTIME_CHECKPOINT="${RUNTIME_CHECKPOINT:-}"
STORAGE_CHECKPOINT="${STORAGE_CHECKPOINT:-}"

EXTRA_ARGS=()
if [[ -n "${RUNTIME_CHECKPOINT}" && -n "${STORAGE_CHECKPOINT}" ]]; then
  echo "RUNTIME_CHECKPOINT and STORAGE_CHECKPOINT are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${STORAGE_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--storage-checkpoint "${STORAGE_CHECKPOINT}")
elif [[ -n "${RUNTIME_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--runtime-checkpoint "${RUNTIME_CHECKPOINT}")
elif [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -z "${RUNTIME_CHECKPOINT}" && -z "${STORAGE_CHECKPOINT}" && "${NO_PRUNE}" == "1" ]]; then
  EXTRA_ARGS+=(--no-prune)
fi

echo "CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR=${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR}"
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

PYTHONPATH=. python scripts/eval_dinov3_vit7b16_cutlass_sparse_nvfp4_accuracy.py \
  --backbone-path "${BACKBONE_PATH}" \
  --head-path "${HEAD_PATH}" \
  --dataset-root "${DATASET_ROOT}" \
  --csv "${CSV}" \
  --zip "${ZIP}" \
  --resize-size "${RESIZE_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --log-interval "${LOG_INTERVAL}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
