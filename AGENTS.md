# AGENTS.md

## Project workflow

每次开始一个较大的开发/修改任务前，我会进行plan，请你将最终决定的plan在 `dev/plans/` 目录下创建计划文件：

- 文件名格式：`序号_xxx_plan.md`
- 序号递增，例如：`001_sparse_reader_plan.md`

每次实现完某个plan，后续的针对这个plan的进一步每一次优化或修改后，在 `dev/impls/序号_xxx_impl.md` 中**追加**开发记录。

开发记录不需要很长，可以包含如下内容（建议可以根据情况添加其他内容）：

```md
## YYYY-MM-DD - 简短标题
- 开发目的
- 修改内容
- 影响文件
- 后续注意
```

## 其他

conda环境：wja-cospaq
cuda：12.8
环境：超算环境，登录节点有网络没有显卡，计算需要提交到计算节点，计算节点没有网络，GPU为RTX 5090。脚本示例如下：
```shell
#!/bin/bash
#SBATCH --job-name=model_test
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/xxx_%j.out
#SBATCH --error=err/xxx_%j.err

echo "Running on $(hostname)"

module load cuda/12.8

# 初始化conda（关键）
source ~/run/miniconda3/etc/profile.d/conda.sh

# 激活环境
conda activate wja-cospaq


export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"

cd /data/home/scxj523/run/wja/project/my/fake/
```