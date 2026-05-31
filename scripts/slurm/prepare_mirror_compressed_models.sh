#!/bin/bash
#SBATCH --job-name=mirror_prep
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/mirror_prep_%j.out
#SBATCH --error=err/mirror_prep_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/

METHODS="${METHODS:-nvfp4 int4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse int4_unstructured_sparse int4_semi_structured_sparse nvfp4_4over6_unstructured_sparse nvfp4_4over6_semi_structured_sparse}"
BENCHMARKS="${BENCHMARKS:-Chameleon GenImage}"
CALIB_SAMPLES="${CALIB_SAMPLES:-64}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LIMIT_ARG=()
if [[ -n "${LIMIT_PER_CLASS:-}" ]]; then
  LIMIT_ARG=(--limit-per-class "${LIMIT_PER_CLASS}")
fi
GENIMAGE_ARG=()
if [[ "${PREFER_EXTRACTED_GENIMAGE:-1}" == "1" ]]; then
  GENIMAGE_ARG=(--prefer-extracted-genimage)
fi

for method in ${METHODS}; do
  echo "[mirror prepare] method=${method}"
  PYTHONPATH=. python scripts/prepare_mirror_compressed_model.py \
    --method "${method}" \
    --benchmarks ${BENCHMARKS} \
    --calib-samples "${CALIB_SAMPLES}" \
    --calib-batch-size "${CALIB_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    "${GENIMAGE_ARG[@]}" \
    "${LIMIT_ARG[@]}"
done
