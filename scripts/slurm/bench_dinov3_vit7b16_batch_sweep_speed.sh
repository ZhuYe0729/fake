#!/bin/bash
#SBATCH --job-name=dinov3_batch_sweep
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_batch_sweep_%j.out
#SBATCH --error=err/dinov3_batch_sweep_%j.err

set -uo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_nvfp4_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_nvfp4_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/results/dinov3_vit7b16_dense artifacts/results/dinov3_vit7b16_cutlass_nvfp4 artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4

METHODS="${METHODS:-dense cutlass_nvfp4 sparse_nvfp4}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
INPUT_SIZE="${INPUT_SIZE:-3 256 256}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m}"
HEAD_PATH="${HEAD_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3_vit7b16_imagenet1k_linear_head/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth}"

DENSE_OUTPUT="${DENSE_OUTPUT:-artifacts/results/dinov3_vit7b16_dense/speed.csv}"
CUTLASS_NVFP4_OUTPUT="${CUTLASS_NVFP4_OUTPUT:-artifacts/results/dinov3_vit7b16_cutlass_nvfp4/speed.csv}"
SPARSE_NVFP4_OUTPUT="${SPARSE_NVFP4_OUTPUT:-artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/speed_storage.csv}"

CUTLASS_NVFP4_RUNTIME_CHECKPOINT="${CUTLASS_NVFP4_RUNTIME_CHECKPOINT:-}"
SPARSE_STORAGE_CHECKPOINT="${SPARSE_STORAGE_CHECKPOINT:-artifacts/checkpoints/dinov3_vit7b16/cutlass_sparse_nvfp4_storage/model.pt}"
SPARSE_RUNTIME_CHECKPOINT="${SPARSE_RUNTIME_CHECKPOINT:-}"

read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
read -r -a METHOD_ARGS <<< "${METHODS}"
read -r -a BATCH_SIZE_ARGS <<< "${BATCH_SIZES}"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"
echo "METHODS=${METHODS}"
echo "BATCH_SIZES=${BATCH_SIZES}"
echo "INPUT_SIZE=${INPUT_SIZE} WARMUP=${WARMUP} ITERS=${ITERS} STOP_ON_FAIL=${STOP_ON_FAIL}"

run_dense() {
  local batch_size="$1"
  PYTHONPATH=. python scripts/bench_dinov3_vit7b16_dense_speed.py \
    --backbone-path "${BACKBONE_PATH}" \
    --head-path "${HEAD_PATH}" \
    --batch-size "${batch_size}" \
    --input-size "${INPUT_SIZE_ARGS[@]}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --output "${DENSE_OUTPUT}"
}

run_cutlass_nvfp4() {
  local batch_size="$1"
  local extra_args=()
  if [[ -n "${CUTLASS_NVFP4_RUNTIME_CHECKPOINT}" ]]; then
    extra_args+=(--runtime-checkpoint "${CUTLASS_NVFP4_RUNTIME_CHECKPOINT}")
  fi
  PYTHONPATH=. python scripts/bench_dinov3_vit7b16_cutlass_nvfp4_speed.py \
    --backbone-path "${BACKBONE_PATH}" \
    --head-path "${HEAD_PATH}" \
    --batch-size "${batch_size}" \
    --input-size "${INPUT_SIZE_ARGS[@]}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --output "${CUTLASS_NVFP4_OUTPUT}" \
    "${extra_args[@]}"
}

run_sparse_nvfp4() {
  local batch_size="$1"
  local extra_args=()
  if [[ -n "${SPARSE_RUNTIME_CHECKPOINT}" ]]; then
    extra_args+=(--runtime-checkpoint "${SPARSE_RUNTIME_CHECKPOINT}")
  elif [[ -n "${SPARSE_STORAGE_CHECKPOINT}" ]]; then
    extra_args+=(--storage-checkpoint "${SPARSE_STORAGE_CHECKPOINT}")
  fi
  PYTHONPATH=. python scripts/bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.py \
    --backbone-path "${BACKBONE_PATH}" \
    --head-path "${HEAD_PATH}" \
    --batch-size "${batch_size}" \
    --input-size "${INPUT_SIZE_ARGS[@]}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --output "${SPARSE_NVFP4_OUTPUT}" \
    "${extra_args[@]}"
}

for method in "${METHOD_ARGS[@]}"; do
  echo "===== method=${method} ====="
  for batch_size in "${BATCH_SIZE_ARGS[@]}"; do
    echo "----- method=${method} batch_size=${batch_size} -----"
    status=0
    case "${method}" in
      dense)
        run_dense "${batch_size}" || status=$?
        ;;
      cutlass_nvfp4)
        run_cutlass_nvfp4 "${batch_size}" || status=$?
        ;;
      sparse_nvfp4)
        run_sparse_nvfp4 "${batch_size}" || status=$?
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 2
        ;;
    esac
    if [[ "${status}" != "0" ]]; then
      echo "FAILED method=${method} batch_size=${batch_size} status=${status}" >&2
      if [[ "${STOP_ON_FAIL}" == "1" ]]; then
        echo "Stop remaining batch sizes for method=${method}" >&2
        break
      fi
    fi
  done
done
