#!/bin/bash
#SBATCH --job-name=dinov3_hybrid_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_hybrid_speed_%j.out
#SBATCH --error=err/dinov3_hybrid_speed_%j.err

set -uo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_nvfp4_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_bf16_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/results/dinov3_vit7b16_cutlass_hybrid

INPUT_SIZE="${INPUT_SIZE:-3 256 256}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
OUTPUT="${OUTPUT:-artifacts/results/dinov3_vit7b16_cutlass_hybrid/speed.csv}"
RUNS="${RUNS:-16:b16_manual 32:b32_manual}"

BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m}"
HEAD_PATH="${HEAD_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3_vit7b16_imagenet1k_linear_head/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth}"

read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
read -r -a RUN_ARGS <<< "${RUNS}"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"
echo "RUNS=${RUNS}"
echo "INPUT_SIZE=${INPUT_SIZE} WARMUP=${WARMUP} ITERS=${ITERS} OUTPUT=${OUTPUT}"

for run_spec in "${RUN_ARGS[@]}"; do
  batch_size="${run_spec%%:*}"
  hybrid_scheme="${run_spec##*:}"
  echo "===== batch_size=${batch_size} hybrid_scheme=${hybrid_scheme} ====="
  PYTHONPATH=. python scripts/bench_dinov3_vit7b16_cutlass_hybrid_speed.py \
    --backbone-path "${BACKBONE_PATH}" \
    --head-path "${HEAD_PATH}" \
    --batch-size "${batch_size}" \
    --hybrid-scheme "${hybrid_scheme}" \
    --input-size "${INPUT_SIZE_ARGS[@]}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --output "${OUTPUT}"
done
