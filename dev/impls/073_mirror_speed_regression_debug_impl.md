## 2026-07-01 - MIRROR 速度异常排查
- 开发目的：在新的 `032_mirror_speed_regression_debug` 目录中定位 batch sweep 中压缩方法速度明显变化的原因。
- 修改内容：修复 `run_batch_speed_sweep.py` 中 dense AMP baseline 强制 BF16 autocast 的口径错误；复测 sparse BF16 连续点和 batch=16 sweep；生成速度异常汇总表与中文诊断报告。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/run_batch_speed_sweep.py`，`artifacts/debug/032_mirror_speed_regression_debug/README.md`，`artifacts/debug/032_mirror_speed_regression_debug/report/diagnosis.md`，`artifacts/debug/032_mirror_speed_regression_debug/results/*`。
- 后续注意：`batch_speed_sweep.py` 的多 backend 长流程会产生不稳定点，尤其 uniform 全量替换和 NVFP4；最终图表应使用单策略/少策略独立进程复测结果，NVFP4 建议多次重复取稳定统计。

## 2026-07-01 - 修正 batch speedup 图
- 开发目的：重新生成 MIRROR 不同 batch size 下未压缩、uniform 方法和 mixed 方法的正确 speedup 柱状图。
- 修改内容：新增独立进程测速脚本，并进一步修正为“每个 batch 固定一张 GPU、该 batch 下所有方法顺序测试”的 by-batch 口径，避免同一组 speedup 使用不同 GPU 或同卡并发。
- 影响文件：`artifacts/debug/032_mirror_speed_regression_debug/scripts/run_corrected_batch_speed_sweep.py`，`artifacts/debug/032_mirror_speed_regression_debug/scripts/run_corrected_batch_speed_sweep_by_batch.py`，`artifacts/debug/032_mirror_speed_regression_debug/corrected_batch_speed_sweep_by_batch/*`，`artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_corrected_by_batch.png`。
- 后续注意：旧的 `batch_speed_sweep_speedup.png` 不应继续使用；正确图使用 `batch_speed_sweep_speedup_corrected_by_batch.png`。

## 2026-07-01 - 充分预热 steady-state batch 图
- 开发目的：按与帕累托图一致的 steady-state 口径，重测不同 batch size 下的 speedup，解决 batch=16 sparse BF16 未充分预热导致的低估。
- 修改内容：将 batch sweep 的压缩策略测速函数统一为 `fake.evaluation.speed.benchmark_forward`；使用 `warmup=100`、`iters=50` 重测；最终采用 sparse-first 顺序，避免先测 NVFP4 后影响 sparse BF16 的稳态速度。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/run_batch_speed_sweep.py`，`artifacts/debug/032_mirror_speed_regression_debug/corrected_batch_speed_sweep_steady_sparse_first_w100_i50/*`，`artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_steady_sparse_first_w100_i50.png`。
- 后续注意：用于汇报的不同 batch 图应使用 `batch_speed_sweep_speedup_steady_sparse_first_w100_i50.png`；该图中 batch=16 的 uniform sparse BF16 为 `1.37x`，与此前帕累托速度量级一致。

## 2026-07-01 - batch 图颜色优化
- 开发目的：提升不同 batch speedup 柱状图在汇报中的可读性。
- 修改内容：将 ours best mixed 柱子改为红色，其余方法改为不同深度灰色，并基于已有 steady-state CSV 重新绘图。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/summarize_batch_speed_sweep.py`，`artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_steady_sparse_first_w100_i50_gray_red.png`。
- 后续注意：PPT 建议使用灰红配色版本，避免多个 uniform 方法同为橙色导致难以区分。

## 2026-07-01 - batch 图精简版
- 开发目的：生成更适合 PPT 展示的精简 batch speedup 图。
- 修改内容：新增精简绘图脚本，仅保留 batch size 8/16/32，去掉 BF16 柱子，图例去掉 `Uniform` 字样，并将我们的方法显示为 `Ours`。
- 影响文件：`artifacts/debug/032_mirror_speed_regression_debug/scripts/plot_batch_speedup_compact.py`，`artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_compact_8_16_32_gray_red.png`。
- 后续注意：精简版图使用已有 steady-state CSV，不涉及重新测速。

## 2026-07-01 - batch 策略选择卡片图
- 开发目的：类似 methods.png 的卡片风格，展示不同 batch size 下我们方法选择的 Linear 后端策略。
- 修改内容：新增策略卡片绘图脚本，基于 batch 8/16/32 的 Ours source policy，按 Attention q/k/v/o、MLP expand gate/up、MLP reduce down 聚合并可视化 Dense BF16、2:4 BF16 和 Dense NVFP4 的比例。
- 影响文件：`artifacts/debug/032_mirror_speed_regression_debug/scripts/plot_batch_strategy_cards.py`，`artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_strategy_cards_8_16_32.png`。
- 后续注意：该图展示的是最终 steady-state batch speedup 图中 Ours 对应的策略，不涉及重新测速。

## 2026-07-01 - batch32 shape-stable 策略修正
- 开发目的：修正 batch=32 Ours 策略中单个 `dense_nvfp4` layer 的 microbench 噪声选择。
- 修改内容：新增 `policy_016_extreme_fastest_shape_stable.json`，将唯一的 `layer15/mlp.gate_proj` 从 `dense_nvfp4` 改回 `sparse_bf16`；按 batch=32 重测该策略，更新 compact speedup 图和策略卡片图。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/speedaware_frontier/policies/policy_016_extreme_fastest_shape_stable.json`，`artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_compact_8_16_32_shape_stable_gray_red.png`，`artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_strategy_cards_8_16_32_shape_stable.png`。
- 后续注意：shape-stable 策略 batch=32 实测 `73.564722 ms`，略快于原 extreme `73.708398 ms`，并消除了不合理的单层 Dense NVFP4 选择。
