## 2026-05-11 - MaxViT 多尺寸评测参数化
- 开发目的：将 MaxViT dense 与压缩评测从 tiny 单模型扩展到 tiny/small/base/large 四个变体。
- 修改内容：新增 MaxViT variant registry；accuracy/speed CLI 支持 `--variant` 和按 config 推断输入尺寸；压缩 checkpoint 生成支持 `--maxvit-variant`；Slurm 脚本支持 `MAXVIT_VARIANT`、variant 结果目录和 large 保守 batch 默认值；README 与结果 summary 标注新目录布局。
- 影响文件：`fake/models/maxvit.py`、`scripts/eval_maxvit_dense_accuracy.py`、`scripts/bench_maxvit_dense_speed.py`、`scripts/prepare_compressed_model.py`、`scripts/slurm/`、`README.md`、`artifacts/results/summary.md`、`dev/plans/005_maxvit_variants_eval_plan.md`。
- 后续注意：历史 `artifacts/results/maxvit_dense/` 和 `artifacts/results/maxvit_compressed/` 不迁移；新增实验以 `maxvit_<variant>_dense/` 和 `maxvit_<variant>_compressed/` 为准。

## 2026-05-11 - Slurm 测试命令清单
- 开发目的：整理所有模型 dense、压缩准备、压缩精度、压缩速度的提交命令，方便批量复制执行。
- 修改内容：新增 `scripts/slurm/all_model_test_commands.md`，按 MaxViT 四个 variant 和 DINOv3 ViT-7B 分组列出命令，并统一带上 `--exclude=wqd10nah09g4`。
- 影响文件：`scripts/slurm/all_model_test_commands.md`、`dev/impls/005_maxvit_variants_eval_impl.md`。
- 后续注意：如果新增模型、方法或节点排除策略，需要同步更新该命令清单。

## 2026-05-11 - 精度结果可视化
- 开发目的：将 dense 与压缩精度结果可视化，便于横向比较各模型与压缩方法。
- 修改内容：新增 `scripts/plot_accuracy_results.py`，为每个模型生成 dense + 五种压缩方法的 Top-1/Top-5 柱状图，并生成全模型 Top-1 汇总热力图。
- 影响文件：`scripts/plot_accuracy_results.py`、各 `artifacts/results/*_compressed/accuracy_comparison.png`、`artifacts/results/accuracy_summary.png`、`dev/impls/005_maxvit_variants_eval_impl.md`。
- 后续注意：当前 tiny 使用历史 `maxvit_dense/` 与 `maxvit_compressed/` 目录；如果后续迁移到 `maxvit_tiny_*`，脚本会优先使用新目录。

## 2026-05-11 - 汇总图数字字号调整
- 开发目的：提高汇总热力图单元格数字可读性。
- 修改内容：将 `accuracy_summary.png` 中的单元格数字字号从 8 调整为 11，并重新生成精度图。
- 影响文件：`scripts/plot_accuracy_results.py`、`artifacts/results/accuracy_summary.png`、`dev/impls/005_maxvit_variants_eval_impl.md`。
- 后续注意：每次重新运行 `scripts/plot_accuracy_results.py` 都会沿用新的字号。
