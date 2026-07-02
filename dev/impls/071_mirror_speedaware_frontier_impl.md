## 2026-07-01 - Speed-Aware Frontier Validation
- 开发目的：根据 MIRROR linear microbench 结果重新设计更符合速度特性的候选策略，并生成更干净的实测帕累托图。
- 修改内容：
  - 修复 `cutlass_wrapper/w4a4_int4.py` 的 sibling package 导入路径，避免 sparse BF16 / dense NVFP4 facade 被 `standalone_w4a4_int4` 导入错误连带阻断。
  - 生成 speed-aware 策略：优先 gate/up dense NVFP4 或 sparse BF16，中高加速段逐步扩展到 MLP sparse BF16，并保留 dense BF16 与 uniform sparse BF16 reference。
  - 重跑 speed validation，确认替换数量非零且速度从约 1.08x 到 1.43x 覆盖。
  - 使用 8 张 GPU 并行完成 10 个策略在 Chameleon + 8 个 GenImage split 上的完整质量验证。
  - 更新汇总图，只显示非支配 speed-aware frontier 与 uniform/reference 点，去掉被支配候选点和多余文字标注。
- 影响文件：
  - `fake/kernels/cutlass/cutlass_wrapper/cutlass_wrapper/w4a4_int4.py`
  - `artifacts/debug/030_mirror_global_pareto/scripts/generate_speedaware_policies.py`
  - `artifacts/debug/030_mirror_global_pareto/scripts/summarize_speedaware_frontier.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/`
- 后续注意：
  - 这批点是基于 microbench 速度结构和已有精度观测的人为 speed-aware 策略，不代表精度加性模型已被彻底修好。
  - 右端高速点仍主要由 uniform sparse BF16 支撑；如果要进一步改善 1.25x 到 1.42x 区间，需要继续研究多层 sparse BF16 的非加性精度退化。

## 2026-07-01 - Near-Uniform Sparse BF16 Points
- 开发目的：补充 `uniform_sparse_bf16` 附近的右侧帕累托点，并在图中显式标出最大速度点。
- 修改内容：
  - 新增 5 个 near-uniform 策略：从全 sparse BF16 出发，将最高 local error 的 attention/MLP/全局若干层恢复为 dense BF16。
  - 追加完成新增策略的速度验证和 Chameleon + GenImage 完整质量验证。
  - 更新汇总图：保留新增非支配右侧前沿点，并用红色星形标出 max-speed measured point。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/generate_speedaware_policies.py`
  - `artifacts/debug/030_mirror_global_pareto/scripts/summarize_speedaware_frontier.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/`
- 后续注意：
  - 新增右侧前沿显示，恢复少量高误差层能显著提高质量但牺牲部分速度；当前最大速度仍为 `uniform_sparse_bf16`。

## 2026-07-01 - Speed And Accuracy Model Diagnostics
- 开发目的：基于当前 15 个 speed-aware / near-uniform 实测策略，重新测试延迟建模和精度建模效果。
- 修改内容：
  - 新增诊断脚本，计算 per-module latency 累加、dense-BF16 normalized latency、affine calibrated latency 与端到端实测 latency 的误差。
  - 计算 additive quality cost 对 CE/NLL 的预测，并和 Chameleon + GenImage 平均 CE/NLL 实测结果对比。
  - 生成新版 `speed_predict.png` 和 `accuracy_predict.png` 到 speed-aware report 目录。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/accuracy_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/model_diagnostics_current.csv`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/model_diagnostics_summary.md`
- 后续注意：
  - 根目录旧图由其他用户/进程所有，当前未覆盖；本次新版图保存在 report 目录。
  - 延迟模型在当前策略集上 raw MAE 约 1.26ms；精度 raw additive NLL MAE 约 0.0509，主要问题仍是 sparse BF16 多层组合下的非加性退化。

## 2026-07-01 - Raw-Only Diagnostic Plot Cleanup
- 开发目的：去掉诊断图中的后验校准曲线/柱子，并额外删除 trimmed 精度图中的 6 个中段失真点。
- 修改内容：
  - `speed_predict` 只保留 measured end-to-end 和 current latency model 两组柱子，并去掉下半图 trend line。
  - `accuracy_predict` 只保留 current additive predicted CE/NLL、measured CE/NLL 和 measured balanced accuracy。
  - `accuracy_predict_trimmed` 删除 6 个中段点：`mlp_sparse_bf16_96`、两个 `mlp_all_plus_attn_sparse_bf16`、`uniform_sparse_bf16`、两个 `restore_worst_attn`。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/accuracy_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/accuracy_predict_trimmed.png`
- 后续注意：
  - summary 仍列出完整 15 点的 raw 诊断表；trimmed 图仅用于展示去除明显失真中段后的可视趋势。

## 2026-07-01 - Diagnostic Plot Axis Cleanup
- 开发目的：去掉诊断图横坐标上的策略名缩写，减少图面干扰。
- 修改内容：`speed_predict`、`accuracy_predict`、`accuracy_predict_trimmed` 均隐藏 x tick labels，仅保留策略点顺序和轴标题。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/accuracy_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/accuracy_predict_trimmed.png`
- 后续注意：如需恢复标签，可以直接从 `model_diagnostics_current.csv` 或 summary 表中对应策略顺序。

## 2026-07-01 - Speed Diagnostic Trim
- 开发目的：让速度诊断图和 trimmed 精度图使用同一批过滤后的策略点，并去掉速度图下半部分散点子图。
- 修改内容：
  - `speed_predict` 改为只绘制过滤后的 9 个策略点。
  - 删除速度诊断图下半部分 predicted-linear-latency scatter 子图，只保留 measured vs current latency model 柱状图。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict.png`
- 后续注意：完整 15 点的数值仍保留在 `model_diagnostics_current.csv` 和 summary 中。

## 2026-07-01 - Method-Type Speed Diagnostic
- 开发目的：将速度诊断图改为按压缩方法分图，并在每个图中按 MIRROR Linear type 展示预测和实测 latency。
- 修改内容：
  - `speed_predict` 改为 2x2 子图，分别对应 `dense_bf16`、`dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4`。
  - 每个子图横轴为 `k_proj/q_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`，纵轴为单 Linear latency，柱子对比 actual microbench 和 latency-model output。
  - 额外输出每个方法的单独图和 `speed_predict_by_method.csv`。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_by_method.csv`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_dense_bf16.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_dense_nvfp4.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_sparse_bf16.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_sparse_nvfp4.png`
- 后续注意：
  - `sparse_nvfp4` 的预测使用 `m=32`，对应 runtime token padding；其余方法使用 batch-16 的 `m=16`。


## 2026-07-01 - Correct Speed Diagnostic Source
- 开发目的：修正 method/type 速度诊断图的对比对象，避免把 CUTLASS kernel predictor 和 MIRROR 当前 optimizer 使用的 module microbench cost table 混在一起比较。
- 修改内容：
  - `speed_predict_by_method` 改为对比 `speed_model/batch_16/module_method_latency.csv` 的 actual module microbench 与 `costs_keyfix_genimage/batch_16/module_method_candidates.csv` 的 optimizer latency table。
  - 重新生成 `speed_predict.png`、各 method 子图和 `speed_predict_by_method.csv`；当前 method/type 级别 MAE 为 0，说明 optimizer 速度表本身就是实测 lookup。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_by_method.csv`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_dense_bf16.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_dense_nvfp4.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_sparse_bf16.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_sparse_nvfp4.png`
- 后续注意：
  - 现在这张图验证的是“当前优化器实际使用的 latency table 是否与单 Linear microbench 一致”。端到端速度误差需要另外看 module latency 加和到整模型 forward 的固定开销、调度开销和组合效应。

## 2026-07-01 - Recheck Kernel Predictor on MIRROR Shapes
- 开发目的：确认 fakevlm/CUTLASS kernel latency predictor 是否能用于 MIRROR 速度建模，以及直接使用时偏差来自哪里。
- 修改内容：
  - 对 MIRROR 的 7 类 Linear type 使用真实 token 展平后的有效 `M=16*201=3216` 重新评估 predictor；`sparse_nvfp4` 使用 32 对齐后的 `M=3232`。
  - 生成 predictor 与 MIRROR module microbench 的逐项对比和 per-method scale 校准摘要。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/speed_model_predictor_recheck/predictor_vs_mirror_microbench.csv`
  - `artifacts/debug/030_mirror_global_pareto/speed_model_predictor_recheck/predictor_recheck_summary.md`
- 后续注意：
  - `M=16` 是错误调用，会导致 dense_bf16 低估 70%-88%。
  - 正确 `M` 后 `dense_bf16`、`dense_nvfp4` 可直接作为初始模型；`sparse_bf16` 和 `sparse_nvfp4` 需要 MIRROR-specific scale 或补充训练点。

## 2026-07-01 - Redraw Speed Diagnostics with Kernel Predictor
- 开发目的：将 `speed_predict` 诊断图从 optimizer lookup table 对比切回真实 kernel latency predictor 对比。
- 修改内容：
  - `speed_predict_by_method.csv` 现在输出 `kernel_latency_predictor` 的 raw prediction，以及按 MIRROR method-level scale 修正后的 prediction。
  - `speed_predict.png` 和各 method 子图改为三组柱子：actual module microbench、kernel latency model、MIRROR-scaled model。
  - MIRROR 有效 `M` 使用 `16*201=3216`；`sparse_nvfp4` 使用 32 对齐后的 `M=3232`。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_by_method.csv`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_dense_bf16.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_dense_nvfp4.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_sparse_bf16.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_sparse_nvfp4.png`
- 后续注意：
  - 当前 scale 是诊断用的轻量校准，不等于重训速度模型；正式 pipeline 可将该 scale 写入 MIRROR speed model 配置或用 MIRROR shapes 补充训练。

## 2026-07-01 - Simplify Speed Prediction Figures
- 开发目的：按最终展示需求，仅保留实测结果和 MIRROR 校准后的速度模型预测结果。
- 修改内容：
  - `speed_predict` 系列图从三组柱子简化为两组柱子：`Measured` 与 `Model predicted`。
  - 图中 MAPE 使用校准后的模型预测误差；raw predictor 数值仍保留在 `speed_predict_by_method.csv` 便于追溯。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_dense_bf16.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_dense_nvfp4.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_sparse_bf16.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speed_predict_sparse_nvfp4.png`
- 后续注意：
  - 当前图例不再使用 `scaled` 表述，但 `Model predicted` 指的是 fakevlm/CUTLASS predictor 经 MIRROR per-method scale 校准后的预测。

## 2026-07-01 - Accuracy Figure Display Cleanup
- 开发目的：按展示需求微调 trimmed 精度诊断图。
- 修改内容：
  - `accuracy_predict_trimmed` 标题去掉 `(trimmed)` 字样。
  - 上半部分 CE/NLL 子图纵轴下限固定为 `0.05`。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speed_accuracy_diagnostics.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/accuracy_predict_trimmed.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/accuracy_predict_trimmed.pdf`
- 后续注意：数据过滤逻辑未改，仅调整展示标题和坐标轴范围。

## 2026-07-01 - Add Report Figure Index
- 开发目的：在 MIRROR debug README 开头标注后续 PPT/汇报主要使用的图片路径。
- 修改内容：
  - 新增中文“汇报用关键图片”小节，列出最终帕累托图、精度建模图、速度建模总览图和 sparse_nvfp4 速度建模图。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/README.md`
- 后续注意：如后续替换最终图，需要同步更新该小节路径。

## 2026-07-01 - Pareto Figure PPT Cleanup
- 开发目的：按 PPT 展示需求精简最终 speed-aware frontier 图。
- 修改内容：
  - 去掉 `speedaware_frontier_clean` 图标题。
  - 横轴改为 `Speedup`。
  - 纵轴改为 `Balanced Accuracy (Chameleon + GenImage mean)`。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/summarize_speedaware_frontier.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speedaware_frontier_clean.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speedaware_frontier_clean.pdf`
- 后续注意：
  - 图中 speedup 仍然是相对 dense default + AMP baseline。

## 2026-07-01 - Speedup Bar Chart
- 开发目的：生成未压缩、各 uniform/reference 方法和本文 speed-aware trade-off 方法的加速比柱状图，便于 PPT 汇报。
- 修改内容：
  - 新增 `build_speedup_bar_uniform_vs_ours.py`，读取已验证速度结果并统一归一化到 `dense_default + AMP` baseline。
  - 输出柱状图 PNG/PDF 和对应 CSV。
  - README 关键图片列表新增该柱状图路径。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_speedup_bar_uniform_vs_ours.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speedup_bar_uniform_vs_ours.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speedup_bar_uniform_vs_ours.pdf`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/speedup_bar_uniform_vs_ours.csv`
  - `artifacts/debug/030_mirror_global_pareto/README.md`
- 后续注意：
  - `Ours trade-off` 当前对应 `gate_up_sparse_bf16_64`，加速比为 1.22x。

## 2026-07-01 - Extreme Fastest Microbench Policy
- 开发目的：基于单 Linear microbench 表生成理论极致速度策略，验证是否有机会超过 uniform sparse_bf16。
- 修改内容：
  - 新增 `build_extreme_fastest_policy.py`，对每个 MIRROR Linear module 在 `dense_bf16/dense_nvfp4/sparse_bf16/sparse_nvfp4` 中选择实测 latency 最小的方法。
  - 生成 `policy_015_extreme_fastest_microbench` 和对应 selected CSV。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/build_extreme_fastest_policy.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/policies/policy_015_extreme_fastest_microbench.json`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/policies/policy_015_extreme_fastest_microbench.csv`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/extreme_fastest_selected.csv`
- 后续注意：
  - 该策略理论 linear latency 比 uniform sparse_bf16 低 2.27%，但端到端是否超过需要 GPU 速度验证。

## 2026-07-01 - Validate Extreme Fastest Speed
- 开发目的：验证 `extreme_fastest_microbench` 策略能否在端到端 forward 中超过 uniform sparse_bf16。
- 修改内容：
  - 使用 `validate_pareto_speed.py` 对 `policy_015_extreme_fastest_microbench` 进行 batch-16 GPU speed validation。
  - 实测结果：`extreme_fastest_microbench` 为 40.299038 ms，speedup vs AMP 为 1.4027x；已有 `uniform_sparse_bf16` 为 39.585003 ms，speedup vs AMP 为 1.4280x。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/extreme_fastest_speed_validation.csv`
- 后续注意：
  - 尽管理论 linear latency 比 uniform sparse_bf16 低 2.27%，端到端反而慢约 1.8%；说明这种混合极致速度策略的调度/组合开销抵消了单 Linear microbench 收益。
