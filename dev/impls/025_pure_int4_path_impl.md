## 2026-05-19 - 纯 INT4 路径接入
- 开发目的：补齐纯 `int4` fake-quant 路径，默认 `group_size=32`，用于和 `nvfp4` 直接对比。
- 修改内容：在压缩 pipeline 中新增 `int4` 方法分支；`prepare/eval/bench` 现有通用脚本可直接接收 `METHOD=int4`；结果汇总脚本增加 `INT4` 的展示项；README 和批量命令清单补充了 MaxViT 四个变体与 DINOv3 的完整命令。
- 影响文件：`fake/compression/pipeline.py`、`scripts/plot_accuracy_results.py`、`scripts/plot_accuracy_compression_speed_summary.py`、`README.md`、`scripts/slurm/all_model_test_commands.md`。
- 后续注意：纯 `int4` 只走 fake quant，不走 SparseGPT；后续若要做真正的 int4 kernel/runtime，还需要单独接入模型加载与算子封装。
