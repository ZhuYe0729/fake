#!/bin/bash
#SBATCH --job-name=llama2_4090_prefill
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/llama2_4090_prefill_%j.out
#SBATCH --error=err/llama2_4090_prefill_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_bf16_ext_${USER}_4090}"
export CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_marlin_nvfp4_ext_${USER}_4090}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err

MODEL_PATH="${MODEL_PATH:-/data/home/scxj523/run/wja/data/models/LLM-Research/llama-2-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/debug/025_llama2_4090_prefill_speed}"
METHODS="${METHODS:-dense_bf16 sparse_bf16 marlin_nvfp4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INPUT_TOKENS="${INPUT_TOKENS:-1024}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
MEASURE_ITERS="${MEASURE_ITERS:-5}"
LINEAR_WARMUP_ITERS="${LINEAR_WARMUP_ITERS:-3}"
LINEAR_MEASURE_ITERS="${LINEAR_MEASURE_ITERS:-10}"
GPU="${GPU:-0}"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

PYTHONPATH=. python artifacts/debug/025_llama2_4090_prefill_speed/scripts/bench_llama2_4090_prefill_speed.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --methods ${METHODS} \
  --batch-size "${BATCH_SIZE}" \
  --input-tokens "${INPUT_TOKENS}" \
  --warmup-iters "${WARMUP_ITERS}" \
  --measure-iters "${MEASURE_ITERS}" \
  --linear-warmup-iters "${LINEAR_WARMUP_ITERS}" \
  --linear-measure-iters "${LINEAR_MEASURE_ITERS}" \
  --gpu "${GPU}"
