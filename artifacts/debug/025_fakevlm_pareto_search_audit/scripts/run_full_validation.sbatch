#!/bin/bash
#SBATCH --job-name=fakevlm_audit
#SBATCH --partition=gpu_5090
#SBATCH --gpus=6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=artifacts/debug/025_fakevlm_pareto_search_audit/logs/slurm_%j.out
#SBATCH --error=artifacts/debug/025_fakevlm_pareto_search_audit/logs/slurm_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8

if [ -f ~/run/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/run/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
else
  echo "Could not find conda.sh" >&2
  exit 1
fi

conda activate wja-cospaq

export HF_HOME="${HF_HOME:-/data/home/scxj523/.cache/huggingface/}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-025-search-audit}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

bash artifacts/debug/025_fakevlm_pareto_search_audit/scripts/run_full_validation.sh
