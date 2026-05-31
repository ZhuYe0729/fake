#!/bin/bash
#SBATCH --job-name=mirror_hybrid_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/mirror_hybrid_speed_%j.out
#SBATCH --error=err/mirror_hybrid_speed_%j.err

set -uo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_nvfp4_ext_${USER}}"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="${CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR:-/tmp/cutlass_wrapper_sparse_bf16_ext_${USER}}"

cd /data/home/scxj523/run/wja/project/my/fake/
mkdir -p out err artifacts/results/mirror_cutlass_hybrid

MODEL_PATH="${MODEL_PATH:-/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight/checkpoint-h-cur.pth}"
MEMORY_PATH="${MEMORY_PATH:-/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight/mirror_phase1.pth}"
BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight/dinov3-huge}"

INPUT_SIZE="${INPUT_SIZE:-3 224 224}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32}"
HYBRID_SCHEMES="${HYBRID_SCHEMES:-dino_b16_like dino_b32_like attn_nvfp4_mlp_bf16 attn_bf16_mlp_nvfp4}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
OUTPUT="${OUTPUT:-artifacts/results/mirror_cutlass_hybrid/speed.csv}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
read -r -a BATCH_SIZE_ARGS <<< "${BATCH_SIZES}"
read -r -a HYBRID_SCHEME_ARGS <<< "${HYBRID_SCHEMES}"

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"
echo "BATCH_SIZES=${BATCH_SIZES}"
echo "HYBRID_SCHEMES=${HYBRID_SCHEMES}"
echo "INPUT_SIZE=${INPUT_SIZE} WARMUP=${WARMUP} ITERS=${ITERS} OUTPUT=${OUTPUT}"

for hybrid_scheme in "${HYBRID_SCHEME_ARGS[@]}"; do
  echo "===== hybrid_scheme=${hybrid_scheme} ====="
  for batch_size in "${BATCH_SIZE_ARGS[@]}"; do
    echo "----- hybrid_scheme=${hybrid_scheme} batch_size=${batch_size} -----"
    status=0
    PYTHONPATH=. python scripts/bench_mirror_cutlass_hybrid_speed.py \
      --model-path "${MODEL_PATH}" \
      --memory-path "${MEMORY_PATH}" \
      --backbone-path "${BACKBONE_PATH}" \
      --batch-size "${batch_size}" \
      --hybrid-scheme "${hybrid_scheme}" \
      --input-size "${INPUT_SIZE_ARGS[@]}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --output "${OUTPUT}" || status=$?
    if [[ "${status}" != "0" ]]; then
      echo "FAILED hybrid_scheme=${hybrid_scheme} batch_size=${batch_size} status=${status}" >&2
      if [[ "${STOP_ON_FAIL}" == "1" ]]; then
        echo "Stop remaining batch sizes for hybrid_scheme=${hybrid_scheme}" >&2
        break
      fi
    fi
  done
done
