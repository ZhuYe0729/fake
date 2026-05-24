# AGENTS.md

## Project workflow

每次开始一个较大的开发/修改任务前，我会进行plan，请你将最终决定的plan在 `dev/plans/` 目录下创建计划文件：

- 文件名格式：`序号_xxx_plan.md`
- 序号递增，例如：`001_sparse_reader_plan.md`

每次实现完某个plan，后续的针对这个plan的进一步每一次优化或修改后，在 `dev/impls/序号_xxx_impl.md` 中**追加**开发记录。我没有进行新的plan时，后续的一些修改和优化默认追加到最新的plan对应的impl中。

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

# 一些额外的基本准则

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.