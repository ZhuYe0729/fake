## 2026-06-23 - Initial cross-workload scaffold
- 开发目的：为 FakeVLM 建立 prefill-only、normal_01、normal_02 三个 workload 下的 uniform 与 linear hybrid 速度对比流程。
- 修改内容：新增计划文件和 debug artifact 脚本目录；实现 E2E 速度测试、4-GPU 任务级 launcher、summary 表生成入口。
- 影响文件：`dev/plans/062_fakevlm_cross_workload_robustness_plan.md`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`，`artifacts/debug/026_fakevlm_cross_workload_robustness/`。
- 后续注意：完整速度数据需要在 GPU 0-3 上运行 full launcher；同一 GPU 不应并发多个速度任务。

## 2026-06-23 - Normal workload smoke fix
- 开发目的：修复 normal_01/normal_02 长输入测速时 finite 检查额外占用显存导致的 OOM。
- 修改内容：将 E2E runner 的 logits finite 检查从完整序列改为只检查最后一个 token logits，避免为 `batch=1,input=16384` 的完整 logits 额外分配 FP32 缓冲。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/run_e2e_speed.py`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
- 后续注意：测速 forward 路径不变；该检查仅用于发现非有限输出。

## 2026-06-23 - Average speedup plot
- 开发目的：为 FakeVLM cross-workload 实验生成一张展示三种 workload 平均结果的柱状图。
- 修改内容：新增 `plot_average_speedup.py`，读取 `workload_method_table.csv` 的 geomean 行绘制各方法平均 speedup，并高亮 `Ours`。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_average_speedup.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_average_geomean_speedup.png`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_average_geomean_speedup.pdf`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
- 后续注意：图中平均采用 speedup 的 geomean；如需 arithmetic mean 可用同脚本 `--average arith_mean` 生成。

## 2026-06-23 - Cross-workload v2 plot
- 开发目的：优化 FakeVLM 展示图，使其更直观体现 cross-workload 稳定优势。
- 修改内容：新增 v2 绘图脚本，删除方法名中的 `Uniform`，移除 `W4A4/W4A16` 方法列，并同时展示 `Prefill`、`Normal-01`、`Normal-02` 和 `Avg.` 四组柱状结果。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v2.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v2_speedup.png`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v2_speedup.pdf`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
- 后续注意：`Avg.` 仍使用 geomean，但图不只展示平均值；标注了 `Ours` 相比 best fixed baseline 的 `+10.3%`。

## 2026-06-23 - Cross-workload v2 label cleanup
- 开发目的：按展示偏好精简 v2 图中文字。
- 修改内容：移除 `+10.3% vs best fixed baseline` 箭头标注，将 `Normal-01`/`Normal-02` 横轴标签改为 `Prefill+Decoding`/`Decode-Heavy`。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v2.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v2_speedup.png`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v2_speedup.pdf`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
- 后续注意：当前 v2 图保留每组 `Ours` 柱顶 speedup 数值，但不再额外标注相对百分比。

## 2026-06-23 - Cross-workload accuracy supplement
- 开发目的：按 FakeVLM uniform 精度评测同口径补充三种 workload 的 `our_linear_hybrid` 精度，并准备速度+精度联合展示图。
- 修改内容：新增 `eval_our_policy_accuracy.py` 复用 020 的 `FakeVLMDataset` 与 `validate` 流程评测 026 的 workload policy；新增 v3 作图脚本，读取 020 uniform 精度、026 速度表和本轮 ours 精度结果生成联合图。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/eval_our_policy_accuracy.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v3_speed_accuracy.py`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
- 后续注意：完整精度评测按 `batch_size=1,max_new_tokens=256,5000 samples` 跑，耗时较长；v3 图需等待三个 `accuracy.json` 齐全后生成。

## 2026-06-24 - Cross-workload accuracy results and v3 plot
- 开发目的：汇总完整精度评测并生成速度与精度联合展示图。
- 修改内容：确认三个 5000-sample 任务完成；结果为 `prefill_only=0.9550`、`normal_01=0.9872`、`normal_02=0.9872`；修正 v3 脚本对嵌套 `global_stats.global_accuracy` 的读取并生成 PNG/PDF。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/accuracy/`，`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v3_speed_accuracy.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v3_speed_accuracy.{png,pdf}`。
- 后续注意：Prefill 策略包含 160 个 sparse BF16 和 64 个 sparse NVFP4 线性层，精度下降至 95.50%；联合展示时应如实说明该速度和精度权衡。

## 2026-06-24 - Merge speed and accuracy axes
- 开发目的：将 v3 图从上下两个子图改为单图双 y 轴展示。
- 修改内容：左轴保留 speedup 柱状图，右轴叠加各方法 accuracy 标记；将 Ours 的 speedup 数值移入柱内，避免与精度标记重叠。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v3_speed_accuracy.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v3_speed_accuracy.{png,pdf}`。

## 2026-06-24 - Refine v3 annotations
- 开发目的：改善双轴图中 Ours 数值和低精度标记的视觉表达。
- 修改内容：将 Ours speedup 数值移动到对应柱子的右上方；不再绘制 `2:4 W4A4` 的 76.86% accuracy 标记。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v3_speed_accuracy.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v3_speed_accuracy.{png,pdf}`。

## 2026-06-24 - Split speed and accuracy figures
- 开发目的：将速度和精度拆分为两张独立图。
- 修改内容：v3 主图仅保留 cross-workload speedup；新增独立 accuracy 分组柱状图，并继续排除 `2:4 W4A4` 精度项；原联合图路径同步覆盖为纯速度图。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v3_speed_accuracy.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v3_speed.{png,pdf}`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v3_accuracy.{png,pdf}`。

## 2026-06-24 - Restore complete accuracy results
- 开发目的：修正独立精度图的方法缺失，并调整速度数值位置。
- 修改内容：恢复 `2:4 W4A4` 的 76.86% 精度结果和完整六方法图例；精度轴调整为 74%–100%；Ours speedup 数值恢复到柱顶居中。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v3_speed_accuracy.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v3_speed.{png,pdf}`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v3_accuracy.{png,pdf}`。

## 2026-06-24 - Use zero-based accuracy axis
- 开发目的：避免截断纵轴放大精度差异。
- 修改内容：将独立精度图纵轴改为 0%–100%，刻度间隔为 20%。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v3_speed_accuracy.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v3_accuracy.{png,pdf}`。

## 2026-06-25 - Cross-workload v4 speed plot
- 开发目的：按展示需求移除 v3 speed 图中的 `Avg.` 分组。
- 修改内容：在 v3 绘图脚本中新增 v4 speed 输出分支，仅绘制 `Prefill`、`Prefill+Decoding`、`Decode-Heavy` 三个 workload。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_cross_workload_v3_speed_accuracy.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v4_speed.{png,pdf}`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
- 后续注意：v3 输出仍保留 `Avg.`，v4 输出专用于去掉平均分组的版本。

## 2026-06-25 - Workload-aware policy case figure
- 开发目的：生成可放在 PPT 右侧竖向区域的策略示意图，展示不同 workload 下不同 layer group 的 hybrid policy 选择。
- 修改内容：新增 9:16 竖版绘图脚本，展示 `Prefill-heavy` 下 Attention/MLP reduce 选择 `2:4 BF16`、MLP expand 选择 `2:4 W4A4`，以及带 decode 的 workload 全部选择 `Hybrid: W4A4 -> W4A16`。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_policy_case_v4.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v4_policy_case.{png,pdf,svg}`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
- 后续注意：该图只展示策略 case，不包含 speed/accuracy；`W4A4 -> W4A16` 在图中作为 hybrid 方法呈现。

## 2026-06-25 - Simplify decode workload policy figure
- 开发目的：避免在带 decode 的两个 workload 中重复展示相同策略的不同 linear type。
- 修改内容：将 `Balanced prefill + decode` 和 `Decode-heavy` 两个 panel 合并为单行 `All language linear layers -> W4A4 -> W4A16`，仅在 `Prefill-heavy` panel 保留 Attention/MLP expand/MLP reduce 的差异。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_policy_case_v4.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v4_policy_case.{png,pdf,svg}`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。

## 2026-06-25 - Policy case figure layout polish
- 开发目的：优化策略示意图排版，使其不强制 9:16 但更适合插入 PPT 右侧区域。
- 修改内容：缩短画布高度、减少 panel 间大面积空白、压实标题和说明文本，并保持 `Prefill-heavy` 三行差异与两个 decode workload 单行 hybrid 策略。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_policy_case_v4.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v4_policy_case.{png,pdf,svg}`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。

## 2026-06-25 - Remove policy figure subtitles
- 开发目的：进一步压缩策略图排版，去掉每个模块标题下方的小字说明。
- 修改内容：删除 `kernel choice follows...` 等 panel subtitle，缩短画布高度并收紧模块内部与模块之间的空隙。
- 影响文件：`artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/plot_policy_case_v4.py`，`artifacts/debug/026_fakevlm_cross_workload_robustness/summary/fakevlm_cross_workload_v4_policy_case.{png,pdf,svg}`，`dev/impls/062_fakevlm_cross_workload_robustness_impl.md`。
