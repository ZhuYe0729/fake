#!/bin/bash
#SBATCH --job-name=dinov3_sparse_bf16_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_sparse_bf16_speed_%j.out
#SBATCH --error=err/dinov3_sparse_bf16_speed_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_bf16_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err

BATCH_SIZE="${BATCH_SIZE:-1}"
INPUT_SIZE="${INPUT_SIZE:-3 256 256}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
CHECKPOINT="${CHECKPOINT:-}"
RUNTIME_CHECKPOINT="${RUNTIME_CHECKPOINT:-}"
OUTPUT="${OUTPUT:-artifacts/results/dinov3_vit7b16_cutlass_sparse_bf16/speed.csv}"

EXTRA_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${RUNTIME_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--runtime-checkpoint "${RUNTIME_CHECKPOINT}")
fi
read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
mkdir -p "$(dirname "${OUTPUT}")"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

PYTHONPATH=. python scripts/bench_dinov3_vit7b16_cutlass_sparse_bf16_speed.py \
  --batch-size "${BATCH_SIZE}" \
  --input-size "${INPUT_SIZE_ARGS[@]}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
