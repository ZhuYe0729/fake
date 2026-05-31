## 2026-05-26 - MIRROR compressed eval scaffolding
- 开发目的：按 029 计划为 MIRROR 增加压缩 checkpoint 准备、Chameleon/GenImage 精度评测和真实 runtime 速度评测入口。
- 修改内容：新增 MIRROR loader helper；压缩模块选择支持 MIRROR backbone transformer Linear；新增 prepare/accuracy/speed Python 脚本和 Slurm 批量脚本；命令文档补充 MIRROR compressed 使用方式；新脚本显式加入仓库根路径以支持直接 `python scripts/...` 启动；Hessian/SparseGPT 校准跳过坏图产生的空 batch。
- 影响文件：`fake/models/mirror.py`、`fake/compression/modules.py`、`fake/compression/pipeline.py`、`scripts/prepare_mirror_compressed_model.py`、`scripts/eval_mirror_compressed_accuracy.py`、`scripts/bench_mirror_compressed_speed.py`、`scripts/slurm/prepare_mirror_compressed_models.sh`、`scripts/slurm/eval_mirror_compressed_accuracy.sh`、`scripts/slurm/bench_mirror_compressed_speed.sh`、`scripts/slurm/all_model_test_commands.md`、`dev/plans/029_mirror_compressed_eval_plan.md`。
- 后续注意：真实评测仍需提交到 RTX 5090 计算节点；速度脚本只接受 dense、CUTLASS dense NVFP4、CUTLASS sparse BF16、CUTLASS sparse NVFP4 四类真实 runtime 方法。

## 2026-05-26 - Fix pure INT4 checkpoint path
- 开发目的：检查 MIRROR prepare 输出时发现纯 `int4` checkpoint 的 quant stats 与 `nvfp4` 完全一致，说明通用 pipeline 仍走 NVFP4 fake quant。
- 修改内容：`compress_model()` 对 `method == "int4"` 改为调用 `fake_quantize_int4_weight()`，保留 `int4_*_sparse` 的 SparseGPT INT4 路径不变。
- 影响文件：`fake/compression/pipeline.py`、`dev/impls/029_mirror_compressed_eval_impl.md`。
- 后续注意：已经生成的 `artifacts/checkpoints/mirror/int4/` 是旧口径，需要单独重跑纯 `int4` prepare；其他 9 个方法无需重跑。

## 2026-05-27 - Summarize and plot MIRROR compressed results
- 开发目的：整理 MIRROR 压缩精度和真实 runtime 速度测试结果，生成可复现的汇总 CSV 与图表。
- 修改内容：新增 `scripts/plot_mirror_compressed_results.py`，按 method/benchmark/dataset 取最新结果，输出 summary、accuracy 总览、GenImage 分项热力图和 speed 总览。
- 影响文件：`scripts/plot_mirror_compressed_results.py`、`artifacts/results/mirror_compressed/summary.csv`、`artifacts/results/mirror_compressed/accuracy_summary.png`、`artifacts/results/mirror_compressed/genimage_breakdown.png`、`artifacts/results/mirror_compressed/speed_summary.png`、`dev/impls/029_mirror_compressed_eval_impl.md`。
- 后续注意：speed 只包含已有真实 runtime 方法；其余 fake-only 方法在 summary 中 speed 字段为空。

## 2026-05-27 - MIRROR runtime batch sweep script
- 开发目的：按 DINO batch sweep 的口径测试 MIRROR dense/CUTLASS runtime 在多个 batch size 下的真实 forward 速度。
- 修改内容：新增 `scripts/slurm/bench_mirror_batch_sweep_speed.sh`，默认跑 `dense nvfp4 semi_structured_sparse nvfp4_semi_structured_sparse` 与 batch size `1 2 4 8 16`，输出 `artifacts/results/mirror_compressed/speed_batch_sweep.csv`。
- 影响文件：`scripts/slurm/bench_mirror_batch_sweep_speed.sh`、`scripts/slurm/all_model_test_commands.md`、`dev/impls/029_mirror_compressed_eval_impl.md`。
- 后续注意：该脚本与 DINO sweep 一样是 random input detector forward-only 口径；只覆盖真实 runtime 方法，不覆盖 fake-only accuracy 方法。

## 2026-05-27 - Plot MIRROR batch sweep speed
- 开发目的：整理 MIRROR 真实 runtime 多 batch size 速度结果，生成吞吐、延迟和相对 dense speedup 曲线。
- 修改内容：新增 `scripts/plot_mirror_batch_sweep_speed.py`，合并各方法 `speed_batch_sweep_*.csv` 最新结果，输出汇总 CSV 与三张 PNG。
- 影响文件：`scripts/plot_mirror_batch_sweep_speed.py`、`artifacts/results/mirror_compressed/speed_batch_sweep_summary.csv`、`artifacts/results/mirror_compressed/speed_batch_sweep_throughput.png`、`artifacts/results/mirror_compressed/speed_batch_sweep_latency.png`、`artifacts/results/mirror_compressed/speed_batch_sweep_speedup.png`、`dev/impls/029_mirror_compressed_eval_impl.md`。
- 后续注意：dense batch size 256 本次 OOM，因此 speedup 曲线只在 dense 有同 batch 结果的点计算。

## 2026-05-27 - Tune MIRROR compressed plots
- 开发目的：按数据集聚合精度总览柱状图，并提升 GenImage 分项热力图的可读性。
- 修改内容：`accuracy_summary.png` 改为每个数据集内并排展示各压缩方法；`genimage_breakdown.png` 改用红黄绿热力配色、放大数字并根据背景亮度自动选择标注颜色。
- 影响文件：`scripts/plot_mirror_compressed_results.py`、`artifacts/results/mirror_compressed/accuracy_summary.png`、`artifacts/results/mirror_compressed/genimage_breakdown.png`、`artifacts/results/mirror_compressed/summary.csv`、`artifacts/results/mirror_compressed/speed_summary.png`、`dev/impls/029_mirror_compressed_eval_impl.md`。
- 后续注意：本次只调整可视化呈现，未改动精度或速度 CSV 的原始数据。

## 2026-05-27 - Hide INT4 methods in MIRROR plots
- 开发目的：当前 MIRROR 可视化结果暂不展示 INT4 相关方法。
- 修改内容：新增绘图专用方法列表，过滤 `int4`、`int4_unstructured_sparse`、`int4_semi_structured_sparse`；重新生成 accuracy summary 与 GenImage breakdown 图。
- 影响文件：`scripts/plot_mirror_compressed_results.py`、`artifacts/results/mirror_compressed/accuracy_summary.png`、`artifacts/results/mirror_compressed/genimage_breakdown.png`、`artifacts/results/mirror_compressed/summary.csv`、`artifacts/results/mirror_compressed/speed_summary.png`、`dev/impls/029_mirror_compressed_eval_impl.md`。
- 后续注意：原始 `accuracy.csv` 和 `summary.csv` 仍保留 INT4 数据，只是不在当前图中显示。

## 2026-05-27 - Label MIRROR accuracy bars
- 开发目的：提升 `accuracy_summary.png` 的信息密度，避免只靠柱高判断数值。
- 修改内容：在每根 accuracy summary 柱顶添加一位小数百分比标注，并放宽 y 轴上限避免高分标注贴边。
- 影响文件：`scripts/plot_mirror_compressed_results.py`、`artifacts/results/mirror_compressed/accuracy_summary.png`、`artifacts/results/mirror_compressed/summary.csv`、`artifacts/results/mirror_compressed/genimage_breakdown.png`、`artifacts/results/mirror_compressed/speed_summary.png`、`dev/impls/029_mirror_compressed_eval_impl.md`。
- 后续注意：本次只改图表标注，不改变指标计算。

## 2026-05-27 - Use horizontal MIRROR bar labels
- 开发目的：按展示需求将 `accuracy_summary.png` 的柱顶数字改为横向显示。
- 修改内容：移除柱顶数值标注的 90 度旋转，并重新生成 MIRROR compressed 可视化图。
- 影响文件：`scripts/plot_mirror_compressed_results.py`、`artifacts/results/mirror_compressed/accuracy_summary.png`、`artifacts/results/mirror_compressed/summary.csv`、`artifacts/results/mirror_compressed/genimage_breakdown.png`、`artifacts/results/mirror_compressed/speed_summary.png`、`dev/impls/029_mirror_compressed_eval_impl.md`。
- 后续注意：标注仍保留一位小数百分比。
