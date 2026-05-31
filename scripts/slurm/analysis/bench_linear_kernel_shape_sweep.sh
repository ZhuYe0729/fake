#!/bin/bash
#SBATCH --job-name=linear_shape_sweep
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/linear_shape_sweep_%j.out
#SBATCH --error=err/linear_shape_sweep_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
export CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_nvfp4_shape_sweep_${USER}}"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_nvfp4_shape_sweep_${USER}}"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_bf16_shape_sweep_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err

FIXED_DIM="${FIXED_DIM:-m}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
SEED="${SEED:-1234}"
FIXED_VALUES="${FIXED_VALUES:-1,16,64,256,4096,16384}"
VARIABLE_VALUES="${VARIABLE_VALUES:-1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384}"
OUTPUT="${OUTPUT:-artifacts/analysis/linear_kernel_shape_sweep/fixed_${FIXED_DIM}.csv}"
RESUME="${RESUME:-1}"

mkdir -p "$(dirname "${OUTPUT}")"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

PYTHONPATH=. python scripts/bench_linear_kernel_shape_sweep.py \
  --fixed-dim "${FIXED_DIM}" \
  --output "${OUTPUT}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --seed "${SEED}" \
  --fixed-values "${FIXED_VALUES}" \
  --variable-values "${VARIABLE_VALUES}" \
  $([[ "${RESUME}" == "1" ]] && echo "--resume")
