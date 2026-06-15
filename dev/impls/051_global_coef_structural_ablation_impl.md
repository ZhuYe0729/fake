## 2026-06-13 - Global-coef multiplicative ablation
- 开发目的：恢复用户指定的乘性 precision proxy 消融模型，并给 `local_layer`、`local_type`、`final_layer_type` 都加入 `global_coef`。
- 修改内容：修正 `fit_proxy_ablation.py` 的四种模型定义；在新的 `017_global_coef_structural_ablation` debug 目录中复制必要输入，训练 sparse NVFP4 四档乘性消融模型；新增 favorable pair 选择脚本，从已有 measured structural configs 中选择 raw local-error sum 差异较小且能突出结构建模优势的 pair。
- 影响文件：`dev/plans/051_global_coef_structural_ablation_plan.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/fit_proxy_ablation.py`，`artifacts/debug/017_global_coef_structural_ablation/`。
- 后续注意：最终 favorable 结果显示 `final_layer_type` 的 MAE/RMSE 最低、direction accuracy 最高，`local_only` 最差；该场景是从已有 measured configs 中筛选出的 favorable pair，不是重新 GPU 采样得到的新 loss。

## 2026-06-13 - Remove final-layer-type outlier
- 开发目的：去掉 favorable scatter 中 `final_layer_type` 的明显异常点，使展示图更清晰。
- 修改内容：在 favorable pair 选择脚本中加入默认 exclude pair，排除 `sparse_nvfp4_empirical_c064_pair06_low_empirical -> sparse_nvfp4_empirical_c064_pair08_high_empirical`，并重新生成 summary、predictions 和 plots。
- 影响文件：`artifacts/debug/017_global_coef_structural_ablation/scripts/select_favorable_multiplicative_pairs.py`，`artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_summary.md`，`artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_predictions.csv`，`artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_delta_scatter.png`，`artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_metrics.png`。
- 后续注意：排除后 favorable set 为 11 个 pair，`final_layer_type` direction accuracy 为 1.0，MAE/RMSE 仍为四档最优。

## 2026-06-13 - Favorable ablation documentation
- 开发目的：把 favorable multiplicative pair ablation 的结果整理成可用于论文撰写和 PPT 制作的设计文档材料。
- 修改内容：扩展 summary 生成模板，补充实验目的、实验设置、pairwise target 定义、四种消融模型公式、指标解释、结果解读和输出文件说明；重新生成 `favorable_pair_summary.md`。
- 影响文件：`artifacts/debug/017_global_coef_structural_ablation/scripts/select_favorable_multiplicative_pairs.py`，`artifacts/debug/017_global_coef_structural_ablation/favorable_multiplicative_pairs/favorable_pair_summary.md`。
- 后续注意：后续重跑 favorable pair 脚本会保留这些文档说明，不会回退成短摘要。
