#!/bin/bash
#SBATCH --job-name=maxvit_nvfp4_micro
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/maxvit_nvfp4_micro_%j.out
#SBATCH --error=err/maxvit_nvfp4_micro_%j.err

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
mkdir -p out err

MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZES="${BATCH_SIZES:-${BATCH_SIZE}}"
if [[ -z "${INPUT_SIZES:-}" ]]; then
  if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
    INPUT_SIZES="3x512x512"
  else
    INPUT_SIZES="3x224x224 3x448x448 3x672x672"
  fi
fi
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
DTYPE="${DTYPE:-bf16}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
QUANT_BACKEND="${QUANT_BACKEND:-cuda}"
OUT_DTYPE="${OUT_DTYPE:-auto}"
SF_LAYOUT="${SF_LAYOUT:-layout_128x4}"
OUTPUT="${OUTPUT:-artifacts/analysis/maxvit_${MAXVIT_VARIANT}/nvfp4/microbench.csv}"
MAX_LAYERS="${MAX_LAYERS:-}"
mkdir -p "$(dirname "${OUTPUT}")"

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

PYTHONPATH=. python scripts/bench_maxvit_nvfp4_micro.py \
  --variant "${MAXVIT_VARIANT}" \
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
