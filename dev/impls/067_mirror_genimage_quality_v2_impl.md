# 067 MIRROR GenImage Quality V2 Implementation

## 2026-06-30 - GenImage V2 calibration and selected validation launch
- 开发目的：用更多策略样本和 GenImage 部分数据 NLL 重新校准 MIRROR 精度模型，降低旧版精度建模导致的被支配策略点。
- 修改内容：新增 V2 policy 生成、精度拟合、cost table 构建、Pareto 优化、validation policy 选择脚本；为 quality validation、validation summary、report plotting 增加自定义输入输出路径参数，避免覆盖旧版结果。
- 运行结果：完成 195 个 GenImage 校准策略评测（195 * 8 split = 1560 rows），V2 精度拟合 RMSE 为 0.0409466；生成 8 个 V2 selected Pareto points；完成 selected speed validation，实测 speedup 约 1.00x 到 2.23x。
- 当前状态：已启动 6 个 GPU 进程并行跑 V2 selected quality validation，输出到 `artifacts/debug/030_mirror_global_pareto/validation_v2_genimage/validation_quality.csv`，日志为 `logs/v2_selected_quality_gpu{0..5}.log`。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/generate_quality_policies_v2.py`、`fit_quality_model_v2.py`、`build_cost_table_v2.py`、`optimize_pareto_v2.py`、`select_validation_policies_v2.py`、`validate_policy_quality.py`、`summarize_validation.py`、`build_report_plots.py`。
- 后续注意：quality validation 完成后需要运行 V2 summarize/report 命令，生成 `validation_v2_genimage/pareto_validation_joined.csv` 和 `report_v2_genimage/final_mirror_report.csv` 及帕累托图。

## 2026-06-30 - GenImage V2 final report
- 开发目的：汇总 V2 selected policies 的端到端 speed/quality validation，并生成最终帕累托图。
- 修改内容：完成 `validation_v2_genimage/validation_quality.csv` 完整性检查；运行 V2 summarize/report，生成 joined CSV、final report、PNG/PDF 图。
- 运行结果：quality validation 完整覆盖 8 个点 * 9 个 split，共 72 rows；最终 report 显示最高实测速度点为 point 022（2.2299x），最高 balanced accuracy 点为 point 021（0.7948）。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/validation_v2_genimage/pareto_validation_joined.csv`、`artifacts/debug/030_mirror_global_pareto/report_v2_genimage/final_mirror_report.csv`、`artifacts/debug/030_mirror_global_pareto/report_v2_genimage/pareto_batch_16_speed_vs_bal_acc.png`、`artifacts/debug/030_mirror_global_pareto/report_v2_genimage/pareto_batch_16_speed_vs_bal_acc.pdf`。
- 后续注意：实测 measured frontier 主要为 point 021 和 point 022；V2 结果中适度压缩反而提升 GenImage/总体 balanced accuracy，后续解释时应区分“建模误差”和“压缩正则化/基线偏置”现象。

## 2026-06-30 - MIRROR checkpoint key normalization fix
- 开发目的：修复 MIRROR dense baseline 精度异常偏低的问题，避免 `strict=False` 静默跳过 backbone fine-tuned 权重。
- 修改内容：加载 checkpoint 时将 `backbone.dino.layer.*` 归一化为当前 transformers 模型使用的 `backbone.dino.model.layer.*`，并在存在 unexpected 或 meaningful missing keys 时直接报错。
- 验证结果：修复后 `load_mirror_dense_detector` 返回 `missing_keys=0, unexpected_keys=0`；少量样本 sanity 中 dense Chameleon bal_acc 为 0.941406，GenImage split bal_acc 基本为 0.988-1.000，恢复到 README 报告量级。
- 影响文件：`fake/models/mirror.py`、`scripts/eval_mirror_dense_accuracy.py`。
- 后续注意：修复前产生的所有 MIRROR quality/NLL、V2 quality model、V2 Pareto report 均应视为无效；速度模型结构不变，通常可保留，但若要严格复现可重测 selected speed。

## 2026-06-30 - Keyfix uniform sanity and GenImage calibration launch
- 开发目的：在 checkpoint key 修复后重新确认 MIRROR 精度评测口径，并重启 GenImage 精度建模样本采集。
- 修改内容：新增 uniform policy 生成脚本；为 local sensitivity 采集脚本增加自定义输出路径和 `--skip-speed`，避免重跑速度微基准；为 V2 policy/fit/cost 脚本增加 keyfix 路径参数。
- 运行结果：完成 5 个 uniform 策略完整 Chameleon+GenImage 评测；dense Chameleon bal_acc=0.946326，GenImage mean bal_acc=0.997849，和 README 量级一致。修复后 local sensitivity 重采完成 1120 rows。
- 当前状态：已启动 8 个 GPU 进程并行采集 `stratified_keyfix_genimage` 的 GenImage 部分样本 NLL，输出到 `artifacts/debug/030_mirror_global_pareto/quality/stratified_keyfix_genimage_quality.csv`。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/generate_uniform_keyfix_policies.py`、`collect_local_errors.py`、`generate_quality_policies_v2.py`、`fit_quality_model_v2.py`、`build_cost_table_v2.py`。
- 后续注意：校准 CSV 预期为 195 policies * 8 GenImage split = 1560 rows；完成后继续 fit keyfix quality model、build cost table、optimize Pareto、selected validation。

## 2026-06-30 - Keyfix final Pareto report
- 开发目的：完成 checkpoint key 修复后的 MIRROR 精度建模、Pareto 求解、selected speed/quality validation 和最终图表。
- 修改内容：基于 `stratified_keyfix_genimage_quality.csv` 和 `sensitivity_keyfix/module_method_local_errors.csv` 拟合 keyfix 精度模型；生成 keyfix cost table、Pareto policies、selected validation、final report 和 PNG/PDF 图。
- 运行结果：GenImage 校准数据完整覆盖 195 policies * 8 split = 1560 rows；selected validation 覆盖 4 points * 9 split = 36 rows。最终 measured frontier 为 dense point 000、point 001、point 018、point 019；最高速度为 point 019（2.1711x, bal_acc 0.9353），point 001 为 1.9933x 且 bal_acc 0.9808。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/global_coefficients_keyfix_genimage/`、`costs_keyfix_genimage/`、`pareto_keyfix_genimage/`、`validation_keyfix_genimage/`、`report_keyfix_genimage/`、`summary/keyfix_genimage_analysis.md`。
- 后续注意：修复前的 `report_v2_genimage` 结果无效；当前有效报告为 `report_keyfix_genimage/final_mirror_report.csv` 和 `report_keyfix_genimage/pareto_batch_16_speed_vs_bal_acc.png`。

## 2026-06-30 - Enhanced Pareto plot with uniform references
- 开发目的：优化 MIRROR keyfix 结果图，加入更多合适策略点和 uniform reference 点。
- 修改内容：新增增强绘图脚本、uniform speed selected 生成脚本、supplemental selected 生成脚本；补充 6 个 sparse_bf16 low-error ratio 策略点；将原优化点、补充点、uniform 点合并到增强版 report/plot。
- 运行结果：5 个 uniform 策略完成端到端 speed validation，6 个 supplemental 策略完成 speed + 完整 Chameleon/GenImage quality validation（6 * 9 = 54 rows）。补充点覆盖 1.92x-2.11x speedup，balanced accuracy 从 0.9913 到 0.9605；uniform_sparse_bf16 为 2.4240x 但 bal_acc=0.9322，作为参考点而非推荐前沿点。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/build_keyfix_enhanced_plot.py`、`prepare_keyfix_uniform_speed_selected.py`、`prepare_keyfix_supplemental_selected.py`、`validation_keyfix_genimage/uniform_speed_validation.csv`、`validation_keyfix_genimage/supplemental_*`、`report_keyfix_genimage_enhanced/`。
- 后续注意：增强图的 uniform speedup 已统一使用实测 dense forward latency 作为分母；`report_keyfix_genimage_enhanced/combined_report.csv` 是当前最完整的可视化输入。

## 2026-06-30 - Dense-scan theoretical frontier validation
- 开发目的：替换人工比例扫描 supplemental 点，改用约束优化理论 Pareto candidate 补充最终图。
- 修改内容：用更密的 quality budget 重新运行 keyfix cost table DP，得到 49 个 unique theoretical Pareto candidates；新增 theoretical selected 生成脚本；选择 13 个均匀分布的理论点完成 speed + 完整 Chameleon/GenImage quality validation；增加增强绘图脚本的 `--no-supplemental` 开关。
- 运行结果：13 个理论点完整覆盖 13 * 9 = 117 行质量结果。全部理论 candidate 报告输出到 `report_keyfix_genimage_theoretical_frontier/`；实测非支配 clean frontier 输出到 `report_keyfix_genimage_theoretical_clean_frontier_enhanced/`，包含 dense reference、8 个理论实测前沿点和 5 个 uniform reference。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/pareto_keyfix_genimage_dense_scan/`、`scripts/prepare_keyfix_theoretical_selected.py`、`validation_keyfix_genimage/theoretical_*`、`report_keyfix_genimage_theoretical_frontier/`、`report_keyfix_genimage_theoretical_frontier_enhanced/`、`report_keyfix_genimage_theoretical_clean_frontier_enhanced/`。
- 后续注意：理论 frontier 是基于精度/速度模型的前沿；最终展示建议优先使用 clean frontier 图，因为它只保留实测后非支配的 theoretical candidates，能避免模型误差导致的被支配点干扰。

## 2026-06-30 - BF16-relative Pareto plot
- 开发目的：补充以全模型 dense_bf16 为 baseline 的 Pareto 可视化，区分 bf16 runtime 本身带来的收益和压缩带来的额外收益。
- 修改内容：基于 clean frontier enhanced 的实测结果重新计算 `speedup_vs_uniform_dense_bf16` 并生成 PNG/PDF 图。
- 运行结果：uniform dense_bf16 baseline latency=52.381904ms；dense fp32 相对 bf16 为 0.533x；理论 clean frontier 主要为 1.04x-1.21x；uniform sparse_bf16 为 1.296x。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/report_keyfix_genimage_theoretical_clean_frontier_bf16_relative/`。
- 后续注意：该图横轴不再表示相对原始 fp32 的总加速，而是表示在全模型 bf16 基线之上的额外 forward speedup。
