#!/bin/bash
#SBATCH --job-name=dump_arch
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dump_arch_%j.out
#SBATCH --error=err/dump_arch_%j.err

echo "Running on $(hostname)"

module load cuda/12.8

# 初始化conda（关键）
source ~/run/miniconda3/etc/profile.d/conda.sh

# 激活环境
conda activate wja-cospaq


export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/

python scripts/temp_dump_models.py --model maxvit
python scripts/temp_dump_models.py --model dinov3
