#!/bin/bash
#SBATCH --job-name=flashinfer_shapes
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/flashinfer_shapes_%j.out
#SBATCH --error=err/flashinfer_shapes_%j.err

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
mkdir -p out err artifacts/analysis/flashinfer

PRESET="${PRESET:-balanced}"
SHAPES="${SHAPES:-}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
DTYPE="${DTYPE:-bf16}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
QUANT_BACKEND="${QUANT_BACKEND:-cuda}"
SF_LAYOUT="${SF_LAYOUT:-layout_128x4}"
OUTPUT="${OUTPUT:-artifacts/analysis/flashinfer/custom_shapes.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/analysis/flashinfer}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

echo "CUDA_MODULE=${CUDA_MODULE}"
echo "FLASHINFER_NO_DOWNLOAD=${FLASHINFER_NO_DOWNLOAD}"
echo "PRESET=${PRESET}"
echo "OUTPUT=${OUTPUT}"
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

EXTRA_ARGS=()
if [[ -n "${SHAPES}" ]]; then
  read -r -a SHAPE_ARGS <<< "${SHAPES}"
  EXTRA_ARGS+=(--shapes "${SHAPE_ARGS[@]}")
fi

PYTHONPATH=. python scripts/bench_flashinfer_custom_shapes.py \
  --preset "${PRESET}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --dtype "${DTYPE}" \
  --gemm-backend "${GEMM_BACKEND}" \
  --quant-backend "${QUANT_BACKEND}" \
  --sf-layout "${SF_LAYOUT}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
  PYTHONPATH=. python scripts/analyze_flashinfer_custom_shapes.py \
    --input "${OUTPUT}" \
    --output-dir "${OUTPUT_DIR}"
fi
