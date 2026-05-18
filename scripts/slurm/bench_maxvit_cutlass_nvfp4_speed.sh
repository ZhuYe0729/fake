#!/bin/bash
#SBATCH --job-name=maxvit_cutlass_nvfp4
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/maxvit_cutlass_nvfp4_%j.out
#SBATCH --error=err/maxvit_cutlass_nvfp4_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_nvfp4_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err

MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
  DEFAULT_BATCH_SIZE=16
else
  DEFAULT_BATCH_SIZE=128
fi
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
INPUT_SIZE="${INPUT_SIZE:-}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-200}"
MODEL_PATH="${MODEL_PATH:-}"
RUNTIME_CHECKPOINT="${RUNTIME_CHECKPOINT:-}"
OUTPUT="${OUTPUT:-artifacts/results/maxvit_${MAXVIT_VARIANT}_cutlass_nvfp4/speed.csv}"

EXTRA_ARGS=()
if [[ -n "${INPUT_SIZE}" ]]; then
  read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
  EXTRA_ARGS+=(--input-size "${INPUT_SIZE_ARGS[@]}")
fi
if [[ -n "${MODEL_PATH}" ]]; then
  EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${RUNTIME_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--runtime-checkpoint "${RUNTIME_CHECKPOINT}")
fi
mkdir -p "$(dirname "${OUTPUT}")"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

PYTHONPATH=. python scripts/bench_maxvit_cutlass_nvfp4_speed.py \
  --variant "${MAXVIT_VARIANT}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
