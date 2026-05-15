## 2026-05-14 - MaxViT NVFP4 microbenchmark

- 开发目的：为 MaxViT tiny 的 FlashInfer NVFP4 路径增加逐层 microbenchmark，拆解 Linear forward、activation quant、FP4 GEMM 和辅助开销。
- 修改内容：新增 `scripts/bench_maxvit_nvfp4_micro.py`，默认覆盖 `3x128x128`、`3x224x224`、`3x384x384`，捕获真实层输入 shape 后逐层输出 CSV；新增 `scripts/README.md` 说明 benchmark 脚本用途和用法。
- 影响文件：`scripts/bench_maxvit_nvfp4_micro.py`、`scripts/README.md`、`dev/plans/007_maxvit_nvfp4_microbenchmark_plan.md`。
- 后续注意：脚本需要在 GPU 计算节点运行；如需正式结果建议增大 `--warmup/--iters`，并在同一节点连续跑 dense/NVFP4 对比以减少波动。

## 2026-05-14 - 简化 scripts README

- 开发目的：将 `scripts/README.md` 调整为通用脚本索引，避免只围绕单次 microbenchmark 任务展开。
- 修改内容：保留 MaxViT/NVFP4 相关脚本的简短说明、microbenchmark 默认输出和快速冒烟命令，删去过长的字段解释和完整运行细节。
- 影响文件：`scripts/README.md`。
- 后续注意：后续新增脚本时继续在 README 中追加短条目，详细设计仍放到 `dev/plans/` 与 `dev/impls/`。

## 2026-05-14 - 新增 microbenchmark analysis Slurm 脚本

- 开发目的：为 MaxViT tiny NVFP4 microbenchmark 增加固定 analysis 输出位置的 Slurm 提交入口。
- 修改内容：新增 `scripts/slurm/analysis/bench_maxvit_nvfp4_micro.sh`，默认输出到 `artifacts/analysis/maxvit_tiny/nvfp4/microbench.csv`，并支持通过环境变量覆盖输入尺寸、batch、warmup/iters、backend、输出路径和快速层数限制。
- 影响文件：`scripts/slurm/analysis/bench_maxvit_nvfp4_micro.sh`、`dev/impls/007_maxvit_nvfp4_microbenchmark_impl.md`。
- 后续注意：正式结果建议使用默认三档输入尺寸；调试时可设置 `MAX_LAYERS=3 WARMUP=5 ITERS=10` 做快速冒烟。

## 2026-05-14 - 修正 MaxViT microbenchmark 默认输入尺寸

- 开发目的：修复 tiny microbenchmark 默认 `3x128x128` 在 MaxViT window partition 中失败的问题。
- 修改内容：默认输入尺寸改为 `3x224x224`、`3x448x448`、`3x672x672`；脚本增加 MaxViT 输入尺寸校验，要求 3 通道、正方形、H/W 为 224 的倍数；同步更新 Slurm analysis 脚本、README 和 plan 中的默认尺寸描述。
- 影响文件：`scripts/bench_maxvit_nvfp4_micro.py`、`scripts/slurm/analysis/bench_maxvit_nvfp4_micro.sh`、`scripts/README.md`、`dev/plans/007_maxvit_nvfp4_microbenchmark_plan.md`。
- 后续注意：如需自定义输入尺寸，优先使用 224 的倍数；否则 timm MaxViT 的 7x7 window partition 会失败。

## 2026-05-14 - 支持多 batch size 与错误记录

- 开发目的：减少多 batch size microbenchmark 的重复提交，并在 OOM 等单个配置失败时保留 CSV 标记。
- 修改内容：`bench_maxvit_nvfp4_micro.py` 新增 `--batch-sizes`，默认 batch size 改为 1；每个 batch/input size 独立执行，成功行写 `status=OK`，失败行写 `status=ERROR`、`error_type`、`error_message`；Slurm analysis 脚本新增 `BATCH_SIZES` 并传给 Python。
- 影响文件：`scripts/bench_maxvit_nvfp4_micro.py`、`scripts/slurm/analysis/bench_maxvit_nvfp4_micro.sh`、`scripts/README.md`。
- 后续注意：CUDA OOM 通常可以继续后续配置，但若遇到更严重的 CUDA context 错误，后续配置可能也会连续 ERROR。

## 2026-05-15 - 修正 MaxViT Large 输入校验

- 开发目的：修复 MaxViT large microbenchmark 被 224 倍数规则误拦截，导致 `3x512x512` 正确输入无法测试的问题。
- 修改内容：`bench_maxvit_nvfp4_micro.py` 改为从模型的 `partition_size` 推导输入尺寸约束；tiny/small/base 仍为 224 倍数，large 为 512 倍数。Slurm analysis 脚本按 `MAXVIT_VARIANT` 设置默认输入和输出目录，large 默认跑 `3x512x512`，输出到 `artifacts/analysis/maxvit_large/nvfp4/microbench.csv`。
- 影响文件：`scripts/bench_maxvit_nvfp4_micro.py`、`scripts/slurm/analysis/bench_maxvit_nvfp4_micro.sh`、`scripts/README.md`、`dev/impls/007_maxvit_nvfp4_microbenchmark_impl.md`。
- 后续注意：`256/768` 对 MaxViT large 仍不是合法 window partition 尺寸；正式补测建议先用 `MAXVIT_VARIANT=large INPUT_SIZES=3x512x512`，大 batch 如 OOM 再分批提交。
