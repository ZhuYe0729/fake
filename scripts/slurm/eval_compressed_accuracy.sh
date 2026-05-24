#!/bin/bash
#SBATCH --job-name=eval_compress
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/eval_compress_%j.out
#SBATCH --error=err/eval_compress_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/

MODEL="${MODEL:-maxvit}"
MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
METHOD="${METHOD:-nvfp4}"
WA_FAKE="${WA_FAKE:-0}"
ACTIVATION_SCALE_RULE="${ACTIVATION_SCALE_RULE:-four_over_six_mse}"
ACTIVATION_QUANT_FORMAT="${ACTIVATION_QUANT_FORMAT:-}"

if [[ "${WA_FAKE}" == "1" ]]; then
  if [[ -z "${ACTIVATION_QUANT_FORMAT}" ]]; then
    if [[ "${METHOD}" == int4* ]]; then
      ACTIVATION_QUANT_FORMAT="int4"
    elif [[ "${METHOD}" == nvfp4* ]]; then
      ACTIVATION_QUANT_FORMAT="nvfp4"
    else
      echo "WA_FAKE=1 requires int4* or nvfp4* METHOD, got METHOD=${METHOD}" >&2
      exit 2
    fi
  fi
  if [[ "${ACTIVATION_QUANT_FORMAT}" == "int4" ]]; then
    ACTIVATION_SCALE_RULE="signed_symmetric"
  elif [[ "${ACTIVATION_QUANT_FORMAT}" != "nvfp4" ]]; then
    echo "Unsupported ACTIVATION_QUANT_FORMAT=${ACTIVATION_QUANT_FORMAT}; expected nvfp4 or int4" >&2
    exit 2
  fi
fi

if [[ "${MODEL}" == "maxvit" ]]; then
  CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/maxvit_${MAXVIT_VARIANT}/${METHOD}/model.pt}"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    SEEDED_CHECKPOINT="$(compgen -G "artifacts/checkpoints/maxvit_${MAXVIT_VARIANT}/${METHOD}_seed*/model.pt" | sort | head -n 1 || true)"
    if [[ -n "${SEEDED_CHECKPOINT}" ]]; then
      CHECKPOINT="${SEEDED_CHECKPOINT}"
    fi
  fi
  if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
    DEFAULT_BATCH_SIZE=16
  else
    DEFAULT_BATCH_SIZE=128
  fi
  BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
  OUTPUT="artifacts/results/maxvit_${MAXVIT_VARIANT}_compressed/accuracy.csv"
  EXTRA_ARGS=()
  if [[ "${WA_FAKE}" == "1" ]]; then
    OUTPUT="artifacts/results/maxvit_${MAXVIT_VARIANT}_compressed/accuracy_wa_fake.csv"
    EXTRA_ARGS+=(--activation-quant --activation-quant-format "${ACTIVATION_QUANT_FORMAT}" --activation-scale-rule "${ACTIVATION_SCALE_RULE}")
  fi
  PYTHONPATH=. python scripts/eval_maxvit_dense_accuracy.py \
    --variant "${MAXVIT_VARIANT}" \
    --batch-size "${BATCH_SIZE}" \
    --checkpoint "${CHECKPOINT}" \
    --method "${METHOD}" \
    --output "${OUTPUT}" \
    "${EXTRA_ARGS[@]}"
elif [[ "${MODEL}" == "dinov3_vit7b16" ]]; then
  CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/${MODEL}/${METHOD}/model.pt}"
  OUTPUT="artifacts/results/dinov3_vit7b16_compressed/accuracy.csv"
  EXTRA_ARGS=()
  if [[ "${WA_FAKE}" == "1" ]]; then
    OUTPUT="artifacts/results/dinov3_vit7b16_compressed/accuracy_wa_fake.csv"
    EXTRA_ARGS+=(--activation-quant --activation-quant-format "${ACTIVATION_QUANT_FORMAT}" --activation-scale-rule "${ACTIVATION_SCALE_RULE}")
  fi
  PYTHONPATH=. python scripts/eval_dinov3_vit7b16_dense_accuracy.py \
    --checkpoint "${CHECKPOINT}" \
    --method "${METHOD}" \
    --output "${OUTPUT}" \
    "${EXTRA_ARGS[@]}"
else
  echo "Unsupported MODEL=${MODEL}" >&2
  exit 1
fi
