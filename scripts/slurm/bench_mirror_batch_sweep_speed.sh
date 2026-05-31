#!/bin/bash
#SBATCH --job-name=mirror_batch_sweep
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/mirror_batch_sweep_%j.out
#SBATCH --error=err/mirror_batch_sweep_%j.err

set -uo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_nvfp4_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_bf16_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_nvfp4_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/results/mirror_compressed

METHODS="${METHODS:-dense nvfp4 semi_structured_sparse nvfp4_semi_structured_sparse}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
INPUT_SIZE="${INPUT_SIZE:-3 224 224}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
OUTPUT="${OUTPUT:-artifacts/results/mirror_compressed/speed_batch_sweep.csv}"

read -r -a METHOD_ARGS <<< "${METHODS}"
read -r -a BATCH_SIZE_ARGS <<< "${BATCH_SIZES}"
read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"
echo "METHODS=${METHODS}"
echo "BATCH_SIZES=${BATCH_SIZES}"
echo "INPUT_SIZE=${INPUT_SIZE} WARMUP=${WARMUP} ITERS=${ITERS} STOP_ON_FAIL=${STOP_ON_FAIL}"
echo "OUTPUT=${OUTPUT}"

for method in "${METHOD_ARGS[@]}"; do
  echo "===== method=${method} ====="
  for batch_size in "${BATCH_SIZE_ARGS[@]}"; do
    echo "----- method=${method} batch_size=${batch_size} -----"
    checkpoint_args=()
    if [[ "${method}" != "dense" ]]; then
      checkpoint_args=(--checkpoint "artifacts/checkpoints/mirror/${method}/model.pt")
    fi
    status=0
    PYTHONPATH=. python scripts/bench_mirror_compressed_speed.py \
      --method "${method}" \
      "${checkpoint_args[@]}" \
      --batch-size "${batch_size}" \
      --input-size "${INPUT_SIZE_ARGS[@]}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --output "${OUTPUT}" || status=$?
    if [[ "${status}" != "0" ]]; then
      echo "FAILED method=${method} batch_size=${batch_size} status=${status}" >&2
      if [[ "${STOP_ON_FAIL}" == "1" ]]; then
        echo "Stop remaining batch sizes for method=${method}" >&2
        break
      fi
    fi
  done
done
