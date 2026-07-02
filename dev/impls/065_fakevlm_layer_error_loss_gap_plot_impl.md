## 2026-06-25 - FakeVLM 层误差与 loss/accuracy gap 可视化
- 开发目的：绘制图展示 FakeVLM 中同一压缩方法下局部误差相近，但最终 loss/accuracy 影响可能差距很大。
- 修改内容：新增 `artifacts/debug/029_fakevlm_layer_error_loss_gap/plot_layer_error_loss_gap.py`，复用 `024_fakevlm_prefill_global_pareto` 的 local error、quality cost、实测 NLL 和实测 accuracy 数据生成 PNG/PDF、summary CSV 和 README。
- 影响文件：`artifacts/debug/029_fakevlm_layer_error_loss_gap/*`，`dev/plans/065_fakevlm_layer_error_loss_gap_plot_plan.md`，`dev/impls/065_fakevlm_layer_error_loss_gap_plot_impl.md`。
- 后续注意：当前图中的 per-module NLL 影响是 024 中用实测整模 NLL 拟合出的 quality-cost proxy，不是重新跑的单层实测 NLL 消融。

## 2026-06-25 - 单独绘制 layer profile 子图
- 开发目的：将四联图中最能体现“同方法局部误差较平滑但 loss proxy 按层尖峰明显”的第二个子图单独导出。
- 修改内容：新增 `plot_layer_profile_only.py`，生成 `fakevlm_layer_profile_error_vs_nll_proxy.png/pdf` 和 `sparse_nvfp4_layer_profile_only.csv`；更新 README 主输出列表。
- 影响文件：`artifacts/debug/029_fakevlm_layer_error_loss_gap/plot_layer_profile_only.py`、`fakevlm_layer_profile_error_vs_nll_proxy.*`、`sparse_nvfp4_layer_profile_only.csv`、`README.md`。
- 后续注意：单图仍采用 `sparse_nvfp4` 的 batch 16 cost table 与 024 拟合 quality-cost proxy 口径。
