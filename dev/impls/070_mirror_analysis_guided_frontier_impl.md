# 070 MIRROR Analysis-Guided Frontier Implementation

## 2026-06-30 - Analysis-guided candidate generation and validation
- 开发目的：不依赖当前不稳定的 additive quality model，基于已有诊断结果人工构造更可能接近真实 Pareto 的 MIRROR 策略，并直接端到端验证。
- 修改内容：新增 `generate_analysis_guided_policies.py` 生成 10 个候选策略；运行真实 forward speed 和完整 Chameleon + GenImage quality；新增 `summarize_analysis_guided_frontier.py` 汇总结果并绘图。
- 运行结果：生成 `analysis_guided_frontier/report/`；新策略在中段前沿明显优于旧点，例如 `lowerr_sparse_bf16_112` 达到 1.1816x vs AMP 且 Bal.Acc=0.98835072，`mlp_lowerr_sparse_bf16_72` 达到 1.2353x 且 Bal.Acc=0.98105630。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/generate_analysis_guided_policies.py`、`summarize_analysis_guided_frontier.py`、`artifacts/debug/030_mirror_global_pareto/analysis_guided_frontier/`。
- 后续注意：当前推荐优先使用低 local-error sparse_bf16 单方法策略；dense_nvfp4 混合点本次不划算，Chameleon 对 sparse 比例非常敏感，需要在正式质量约束中单独报告。

## 2026-06-30 - Clean frontier plot with uniform references
- 开发目的：优化 combined measured frontier 图，减少无关文字和 dominated 点，并加入 uniform reference。
- 修改内容：在 `summarize_analysis_guided_frontier.py` 中新增 clean combined plot 生成逻辑，只展示 previous frontier、analysis-guided frontier、uniform references 和 measured frontier，纵轴范围改为 0.85 起。
- 运行结果：生成 `analysis_guided_frontier/report/combined_measured_frontier_clean.png` 和 `.pdf`。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/summarize_analysis_guided_frontier.py`、`artifacts/debug/030_mirror_global_pareto/analysis_guided_frontier/report/combined_measured_frontier_clean.*`。
- 后续注意：原始 `combined_measured_frontier.png` 仍保留；论文/汇报优先使用 clean 版本。
