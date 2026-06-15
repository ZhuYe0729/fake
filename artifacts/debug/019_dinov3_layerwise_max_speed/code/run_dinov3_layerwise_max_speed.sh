#!/bin/bash
#SBATCH --job-name=dinov3_layerwise_max_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_layerwise_max_speed_%j.out
#SBATCH --error=err/dinov3_layerwise_max_speed_%j.err

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p out err artifacts/debug/019_dinov3_layerwise_max_speed

BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64 128}"
INPUT_SIZE="${INPUT_SIZE:-3 256 256}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/debug/019_dinov3_layerwise_max_speed}"
BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m}"
HEAD_PATH="${HEAD_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3_vit7b16_imagenet1k_linear_head/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth}"
MODEL_ROOT="${MODEL_ROOT:-fake/kernels/cutlass/cutlass_wrapper/artifacts/modeling}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"

read -r -a BATCH_ARGS <<< "${BATCH_SIZES}"
read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
EXTRA_ARGS=()
if [[ "${GENERATE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--generate-only)
fi

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"
echo "BATCH_SIZES=${BATCH_SIZES}"
echo "INPUT_SIZE=${INPUT_SIZE} WARMUP=${WARMUP} ITERS=${ITERS} OUTPUT_ROOT=${OUTPUT_ROOT}"

PYTHONPATH=. python artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.py \
  --backbone-path "${BACKBONE_PATH}" \
  --head-path "${HEAD_PATH}" \
  --batch-sizes "${BATCH_ARGS[@]}" \
  --input-size "${INPUT_SIZE_ARGS[@]}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  "${EXTRA_ARGS[@]}"
