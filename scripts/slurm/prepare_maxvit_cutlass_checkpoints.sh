#!/bin/bash
#SBATCH --job-name=prep_maxvit_cutlass
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/prep_maxvit_cutlass_%j.out
#SBATCH --error=err/prep_maxvit_cutlass_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_nvfp4_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_nvfp4_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_bf16_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/checkpoints

MAXVIT_VARIANTS="${MAXVIT_VARIANTS:-tiny small base large}"
BACKENDS="${BACKENDS:-dense_nvfp4 sparse_nvfp4 sparse_bf16}"
MODEL_PATH="${MODEL_PATH:-}"
CHECKPOINT="${CHECKPOINT:-}"
SPARSE_BF16_CHECKPOINT="${SPARSE_BF16_CHECKPOINT:-}"
NO_PRUNE="${NO_PRUNE:-0}"

read -r -a VARIANT_ARGS <<< "${MAXVIT_VARIANTS}"
read -r -a BACKEND_ARGS <<< "${BACKENDS}"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"

for variant in "${VARIANT_ARGS[@]}"; do
  for backend in "${BACKEND_ARGS[@]}"; do
    EXTRA_ARGS=()
    if [[ -n "${MODEL_PATH}" ]]; then
      EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
    fi
    if [[ "${backend}" == "sparse_nvfp4" && -n "${CHECKPOINT}" ]]; then
      EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
    fi
    if [[ "${backend}" == "sparse_bf16" && -n "${SPARSE_BF16_CHECKPOINT}" ]]; then
      EXTRA_ARGS+=(--checkpoint "${SPARSE_BF16_CHECKPOINT}")
    fi
    if [[ "${backend}" == "sparse_bf16" && -z "${SPARSE_BF16_CHECKPOINT}" ]]; then
      DEFAULT_SPARSE_BF16_CHECKPOINT="artifacts/checkpoints/maxvit_${variant}/semi_structured_sparse/model.pt"
      if [[ "${variant}" == "tiny" && ! -f "${DEFAULT_SPARSE_BF16_CHECKPOINT}" ]]; then
        DEFAULT_SPARSE_BF16_CHECKPOINT="artifacts/checkpoints/maxvit/semi_structured_sparse/model.pt"
      fi
      if [[ -f "${DEFAULT_SPARSE_BF16_CHECKPOINT}" ]]; then
        EXTRA_ARGS+=(--checkpoint "${DEFAULT_SPARSE_BF16_CHECKPOINT}" --no-prune)
      fi
    fi
    if [[ "${backend}" == "sparse_nvfp4" && "${NO_PRUNE}" == "1" ]]; then
      EXTRA_ARGS+=(--no-prune)
    fi
    if [[ "${backend}" == "sparse_bf16" && "${NO_PRUNE}" == "1" ]]; then
      EXTRA_ARGS+=(--no-prune)
    fi
    PYTHONPATH=. python scripts/prepare_maxvit_cutlass_checkpoints.py \
      --variant "${variant}" \
      --backend "${backend}" \
      "${EXTRA_ARGS[@]}"
  done
done
