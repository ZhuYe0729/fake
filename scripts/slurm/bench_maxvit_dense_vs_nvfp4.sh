#!/bin/bash
#SBATCH --job-name=maxvit_dv4
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/maxvit_dv4_%j.out
#SBATCH --error=err/maxvit_dv4_%j.err

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

MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
  DEFAULT_BATCH_SIZE=16
else
  DEFAULT_BATCH_SIZE=128
fi
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-200}"
DENSE_DTYPE="${DENSE_DTYPE:-bf16}"
NVFP4_DTYPE="${NVFP4_DTYPE:-bf16}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
QUANT_BACKEND="${QUANT_BACKEND:-cuda}"

echo "CUDA_MODULE=${CUDA_MODULE}"
echo "FLASHINFER_NO_DOWNLOAD=${FLASHINFER_NO_DOWNLOAD}"
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

echo "Benchmark dense MaxViT"
PYTHONPATH=. python scripts/bench_maxvit_dense_speed.py \
  --variant "${MAXVIT_VARIANT}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --dtype "${DENSE_DTYPE}" \
  --output "artifacts/results/maxvit_${MAXVIT_VARIANT}_dense/speed.csv"

echo "Benchmark FlashInfer NVFP4 MaxViT"
PYTHONPATH=. python scripts/bench_maxvit_nvfp4_speed.py \
  --variant "${MAXVIT_VARIANT}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --dtype "${NVFP4_DTYPE}" \
  --gemm-backend "${GEMM_BACKEND}" \
  --quant-backend "${QUANT_BACKEND}" \
  --output "artifacts/results/maxvit_${MAXVIT_VARIANT}_nvfp4/speed.csv"
