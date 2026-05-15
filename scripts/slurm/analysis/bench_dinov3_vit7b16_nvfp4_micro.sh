#!/bin/bash
#SBATCH --job-name=dinov3_nvfp4_micro
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_nvfp4_micro_%j.out
#SBATCH --error=err/dinov3_nvfp4_micro_%j.err

set -euo pipefail

echo "Running on $(hostname)"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
module load "${CUDA_MODULE}"
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export FLASHINFER_NO_DOWNLOAD="${FLASHINFER_NO_DOWNLOAD:-1}"
export CUDA_MODULE

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/analysis/dinov3_vit7b16/nvfp4

BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZES="${BATCH_SIZES:-${BATCH_SIZE}}"
INPUT_SIZES="${INPUT_SIZES:-3x128x128 3x256x256 3x384x384}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
DTYPE="${DTYPE:-bf16}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
QUANT_BACKEND="${QUANT_BACKEND:-cuda}"
OUT_DTYPE="${OUT_DTYPE:-auto}"
SF_LAYOUT="${SF_LAYOUT:-layout_128x4}"
OUTPUT="${OUTPUT:-artifacts/analysis/dinov3_vit7b16/nvfp4/microbench.csv}"
MAX_LAYERS="${MAX_LAYERS:-}"
BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m}"
HEAD_PATH="${HEAD_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3_vit7b16_imagenet1k_linear_head/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth}"

echo "CUDA_MODULE=${CUDA_MODULE}"
echo "FLASHINFER_NO_DOWNLOAD=${FLASHINFER_NO_DOWNLOAD}"
echo "OUTPUT=${OUTPUT}"
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZES}"
read -r -a BATCH_SIZE_ARGS <<< "${BATCH_SIZES}"
EXTRA_ARGS=()
if [[ -n "${MAX_LAYERS}" ]]; then
  EXTRA_ARGS+=(--max-layers "${MAX_LAYERS}")
fi

PYTHONPATH=. python scripts/bench_dinov3_vit7b16_nvfp4_micro.py \
  --backbone-path "${BACKBONE_PATH}" \
  --head-path "${HEAD_PATH}" \
  --batch-sizes "${BATCH_SIZE_ARGS[@]}" \
  --input-sizes "${INPUT_SIZE_ARGS[@]}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --dtype "${DTYPE}" \
  --gemm-backend "${GEMM_BACKEND}" \
  --quant-backend "${QUANT_BACKEND}" \
  --out-dtype "${OUT_DTYPE}" \
  --sf-layout "${SF_LAYOUT}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
