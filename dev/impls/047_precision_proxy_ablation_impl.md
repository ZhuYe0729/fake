## 2026-06-13 - Precision proxy ablation scaffold
- 开发目的：对 `sparse_bf16`、`dense_nvfp4`、`sparse_nvfp4` 的精度代理做结构消融，验证 local error、layer depth、linear type 的设计价值。
- 修改内容：新增 `fit_proxy_ablation.py`，支持 `local_only`、`local_layer`、`local_type`、`final_layer_type` 四种结构，并将 dense 校准作为参考行。
- 影响文件：`dev/plans/047_precision_proxy_ablation_plan.md`，`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/fit_proxy_ablation.py`。
- 后续注意：该实验复用已有 120-policy loss 和 local error 数据，不需要重新跑 GPU loss。

## 2026-06-13 - Ablation results generated
- 开发目的：生成三种方法的结构消融表和图。
- 修改内容：运行离线 ablation，输出 metrics、predictions、coefficients、summary 和 holdout Spearman/RMSE 图；修正 `local_type` 消融的尺度归一化问题。
- 影响文件：`artifacts/debug/016_llama2_sparse_bf16_precision_proxy/ablation/`，`scripts/fit_proxy_ablation.py`。
- 后续注意：final 结构在 MAE/RMSE 上整体略优；排序指标差异较小，说明当前 sampled policies 的主信号仍主要来自压缩数量/local error 总量。

## 2026-06-13 - Holdout prediction scatter plots
- 开发目的：直观看不同消融结构的 holdout loss 预测情况，解释纯 local 相关性较高的原因。
- 修改内容：新增每个方法的 holdout predicted-vs-measured scatter grid，并按 `selected_modules` 上色。
- 影响文件：`scripts/fit_proxy_ablation.py`，`ablation/proxy_ablation_holdout_predictions_*.png`，`ablation/proxy_ablation_summary.md`。
- 后续注意：图中颜色分层明显，说明纯 local 主要抓住了压缩数量/local error 总量的强趋势；同一压缩数量内部排序提升仍需要更有针对性的采样来验证。
