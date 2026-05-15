#!/bin/bash
#SBATCH --job-name=check_fi_nvfp4
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/check_fi_nvfp4_%j.out
#SBATCH --error=err/check_fi_nvfp4_%j.err

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

M="${M:-256}"
N="${N:-512}"
K="${K:-1024}"
ITERS="${ITERS:-50}"
DTYPE="${DTYPE:-bf16}"
GEMM_BACKEND="${GEMM_BACKEND:-auto}"
QUANT_BACKEND="${QUANT_BACKEND:-cuda}"

echo "CUDA_MODULE=${CUDA_MODULE}"
echo "FLASHINFER_NO_DOWNLOAD=${FLASHINFER_NO_DOWNLOAD}"
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

PYTHONPATH=. python scripts/check_flashinfer_nvfp4.py \
  --m "${M}" \
  --n "${N}" \
  --k "${K}" \
  --iters "${ITERS}" \
  --dtype "${DTYPE}" \
  --backend "${GEMM_BACKEND}" \
  --quant-backend "${QUANT_BACKEND}"
