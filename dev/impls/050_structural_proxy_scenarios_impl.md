## 2026-06-13 - Sparse NVFP4 structural scenario design
- 开发目的：设计突出 layer depth / linear type 影响、弱化 local error sum 的 sparse NVFP4 特定场景。
- 修改内容：新增基于现有 fitted structural proxy 的场景生成与真实 loss 分析；验证该方向在真实 loss 上反向，不能作为展示样例。新增基于 stratified 数据估计 empirical residual effect 的场景生成脚本，得到 raw local sum 高度匹配但 early/down/gate 与 late/q/k/v/o 组成明显不同的 pairs。
- 影响文件：`dev/plans/050_structural_proxy_scenarios_plan.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/design_sparse_nvfp4_structural_scenarios.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/analyze_sparse_nvfp4_structural_scenarios.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/design_sparse_nvfp4_empirical_scenarios.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/analyze_sparse_nvfp4_empirical_scenarios.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/`。
- 后续注意：empirical structural pairs 已生成但真实 GPU loss 因当前会话 usage limit 未能启动；恢复额度后应运行 `output-tag=empirical_structural` 的 sparse NVFP4 loss，再补 pairwise 分析。

## 2026-06-13 - Empirical structural scenario validation
- 开发目的：验证 empirical residual effect 设计的 sparse NVFP4 structural scenarios 是否能作为 layer/type 消融展示场景。
- 修改内容：完成 24 条 empirical structural sparse NVFP4 真实 loss；生成 pairwise summary 和 loss delta 图。
- 影响文件：`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/loss/loss_samples_sparse_nvfp4_empirical_structural.csv`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_scenario_results.csv`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_scenario_loss_summary.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_scenario_loss_delta.png`。
- 后续注意：该场景在 raw local sum 几乎匹配的情况下 12/12 high-risk 组合 loss 更高，适合用于证明仅 local error sum 不足、layer/type 结构影响真实存在。

## 2026-06-13 - Balanced calibrated ablation summary
- 开发目的：把 sparse NVFP4 structural scenario 结果整理成用户要求的四档消融：`local_only`、`local_depth`、`local_type`、`final_depth_type`。
- 修改内容：基于 balanced empirical scenarios 的真实 loss，新增 scale-only calibrated ablation 汇总脚本，输出消融表、预测明细和指标图；报告中显式写出 MAE / direction 排名。
- 影响文件：`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/calibrate_sparse_nvfp4_empirical_ablation.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_calibrated_ablation_summary.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_calibrated_ablation_summary.csv`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_calibrated_ablation_predictions.csv`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_calibrated_ablation_metrics.png`。
- 后续注意：该消融在同一组 scale-only calibration 下显示 `final_depth_type` 的 MAE 最优，`local_only` 的 MAE/RMSE 和方向准确率最差；RMSE 上 `local_type` 略优于 final，报告时应以 MAE 和方向准确率作为主指标。

## 2026-06-13 - Config-level loss prediction plot
- 开发目的：补充一张基于测试 loss 和四种消融代理预测 loss 的直观 scatter 图，解释 pairwise 指标之外的 config-level 行为。
- 修改内容：新增 config-level loss prediction plotting 脚本，使用 stratified sparse NVFP4 样本拟合四种代理，在 balanced structural configs 上输出 measured loss delta vs predicted loss delta 的 scatter、指标表和预测明细。
- 影响文件：`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/plot_sparse_nvfp4_empirical_ablation_loss_predictions.py`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_config_loss_ablation_summary.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_config_loss_ablation_predictions.csv`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_config_loss_ablation_scatter.png`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/structural_scenarios/sparse_nvfp4_empirical_balanced_config_loss_ablation_metrics.png`。
- 后续注意：config-level 图显示 `local_only` 预测几乎不区分测试配置，depth/type 结构特征显著提高相关性；但绝对 loss delta 上 `local_type` 优于 `final_depth_type`，因此 final 最优的主展示仍应使用 controlled pairwise delta 消融。
