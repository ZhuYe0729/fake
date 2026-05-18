#!/bin/bash
#SBATCH --job-name=dinov3_sparse_nvfp4_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_sparse_nvfp4_speed_%j.out
#SBATCH --error=err/dinov3_sparse_nvfp4_speed_%j.err

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
INPUT_SIZE="${INPUT_SIZE:-3 256 256}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
OUTPUT="${OUTPUT:-artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/speed.csv}"
BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m}"
HEAD_PATH="${HEAD_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3_vit7b16_imagenet1k_linear_head/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth}"
CHECKPOINT="${CHECKPOINT:-}"
NO_PRUNE="${NO_PRUNE:-0}"
RUNTIME_CHECKPOINT="${RUNTIME_CHECKPOINT:-}"
STORAGE_CHECKPOINT="${STORAGE_CHECKPOINT:-}"

read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
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

PYTHONPATH=. python scripts/bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.py \
  --backbone-path "${BACKBONE_PATH}" \
  --head-path "${HEAD_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --input-size "${INPUT_SIZE_ARGS[@]}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
