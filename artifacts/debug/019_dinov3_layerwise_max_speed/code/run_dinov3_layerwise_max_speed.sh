#!/bin/bash
#SBATCH --job-name=dinov3_layerwise_max_speed
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_layerwise_max_speed_%j.out
#SBATCH --error=err/dinov3_layerwise_max_speed_%j.err

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

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/scxj523/run/wja/project/my/fake}"
if [[ -d "${PROJECT_ROOT}/fake" && -f "${PROJECT_ROOT}/artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.py" ]]; then
  REPO_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/fake" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
export REPO_ROOT
cd "${REPO_ROOT}"
mkdir -p out err 2>/dev/null || true

BATCH_SIZES="${BATCH_SIZES:-32}"
INPUT_SIZE="${INPUT_SIZE:-3 256 256}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
BACKBONE_PATH="${BACKBONE_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m}"
HEAD_PATH="${HEAD_PATH:-/data/home/scxj523/run/wja/data/models/facebook/dinov3_vit7b16_imagenet1k_linear_head/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth}"
MODEL_ROOT="${MODEL_ROOT:-fake/kernels/cutlass/cutlass_wrapper/artifacts/modeling}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"
RUN_ACCURACY="${RUN_ACCURACY:-1}"
ACCURACY_BATCH_SIZE="${ACCURACY_BATCH_SIZE:-1}"
ACCURACY_NUM_WORKERS="${ACCURACY_NUM_WORKERS:-4}"
ACCURACY_LOG_INTERVAL="${ACCURACY_LOG_INTERVAL:-50}"
HYBRID_SCHEME="${HYBRID_SCHEME:-b32_manual}"
DATASET_ROOT="${DATASET_ROOT:-/data/home/scxj523/run/wja/data/datasets/imagenet_val}"
DATASET_CSV="${DATASET_CSV:-val.csv}"
DATASET_ZIP="${DATASET_ZIP:-imagenet_val.zip}"

if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$(PYTHONPATH=. python - <<'PY'
from pathlib import Path
import os
repo = Path(os.environ["REPO_ROOT"])
candidates = [
    repo / "artifacts/debug/019_dinov3_layerwise_max_speed",
]
submit_dir = os.environ.get("SLURM_SUBMIT_DIR")
if submit_dir:
    candidates.append(Path(submit_dir) / "artifacts/debug/019_dinov3_layerwise_max_speed")
home = os.environ.get("HOME")
if home:
    candidates.append(Path(home) / "dinov3_layerwise_max_speed_019")
for candidate in candidates:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError:
        continue
    if os.access(candidate, os.W_OK):
        print(candidate)
        break
else:
    raise SystemExit("No writable OUTPUT_ROOT candidate found")
PY
)"
else
  mkdir -p "${OUTPUT_ROOT}"
fi

read -r -a BATCH_ARGS <<< "${BATCH_SIZES}"
read -r -a INPUT_SIZE_ARGS <<< "${INPUT_SIZE}"
EXTRA_ARGS=()
if [[ "${GENERATE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--generate-only)
fi

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'no cuda')"
echo "BATCH_SIZES=${BATCH_SIZES}"
echo "INPUT_SIZE=${INPUT_SIZE} WARMUP=${WARMUP} ITERS=${ITERS} OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_ACCURACY=${RUN_ACCURACY} HYBRID_SCHEME=${HYBRID_SCHEME} ACCURACY_BATCH_SIZE=${ACCURACY_BATCH_SIZE}"

PYTHONPATH=. python "${REPO_ROOT}/artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.py" \
  --backbone-path "${BACKBONE_PATH}" \
  --head-path "${HEAD_PATH}" \
  --batch-sizes "${BATCH_ARGS[@]}" \
  --input-size "${INPUT_SIZE_ARGS[@]}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --output-root "${OUTPUT_ROOT}" \
  --model-root "${MODEL_ROOT}" \
  "${EXTRA_ARGS[@]}"

if [[ "${RUN_ACCURACY}" == "1" ]]; then
  PYTHONPATH=. python "${REPO_ROOT}/artifacts/debug/019_dinov3_layerwise_max_speed/code/eval_dinov3_hybrid_accuracy.py" \
    --backbone-path "${BACKBONE_PATH}" \
    --head-path "${HEAD_PATH}" \
    --dataset-root "${DATASET_ROOT}" \
    --csv "${DATASET_CSV}" \
    --zip "${DATASET_ZIP}" \
    --batch-size "${ACCURACY_BATCH_SIZE}" \
    --num-workers "${ACCURACY_NUM_WORKERS}" \
    --log-interval "${ACCURACY_LOG_INTERVAL}" \
    --hybrid-scheme "${HYBRID_SCHEME}" \
    --output "${OUTPUT_ROOT}/hybrid_accuracy.csv"
fi
