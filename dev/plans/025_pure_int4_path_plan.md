## 2026-05-19 - 纯 INT4 路径支持
- 开发目的：新增纯 `int4` 路径，默认 `group_size=32`，用于和现有 `nvfp4` 做直接对比。
- 修改内容：在压缩 pipeline 中注册 `int4` 纯 fake-quant 分支，保留现有 `int4_*_sparse` 路径不变；同步更新结果汇总脚本与文档中的可用方法和命令示例。
- 影响文件：`fake/compression/pipeline.py`、`fake/compression/int4.py`、`scripts/plot_accuracy_results.py`、`scripts/plot_accuracy_compression_speed_summary.py`、`README.md`、`scripts/slurm/all_model_test_commands.md`。
- 后续注意：纯 `int4` 不走 SparseGPT，也不需要稀疏 mask；默认 checkpoint 路径与 `nvfp4` 保持同一套 compressed 目录结构。
