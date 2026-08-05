# 136 Quality Modeling Insight Figure Plan

## 目标

基于已有 Llama-2-7B kernel-aware 单 Linear 消融结果，制作可直接用于论文
Figure 3 的 quality-modeling insight 图，说明 local output error 不能单独决定整模
quality impact，且 sensitivity 与 depth 和 projection type 有关。

## 假设与口径

- 使用 `015_llama2_prefill_kernel_loss_modeling` 中真实 CUTLASS
  `sparse_nvfp4` kernel 的已有结果，不运行新的 GPU 实验。
- local error 使用 `output_rel_mse`。
- global quality impact 使用只压缩一个 Linear 后实测的 WikiText-2
  `loss_delta_vs_dense`，统一记为 measured single-Linear
  \(\Delta\mathrm{NLL}\)。
- 图中保留旧实验的七类原始 projection：
  `q/k/v/o/gate/up/down_proj`，不将其描述成当前 runtime 的四类 fused action。
- 该图作为结构性设计观察；当前 Llama-3.1 mixed-policy holdout ablation 继续作为
  structure/phase-aware quality model 的主要验证。

## 实施步骤

1. 新建隔离的 figure artifact 目录和可复现绘图脚本。
   - 验证：脚本只读取已有 joined CSV，筛选 224 条 `sparse_nvfp4`
     单 Linear 数据。
2. 绘制双子图。
   - (a) local output relative MSE 与 measured single-Linear
     \(\Delta\mathrm{NLL}\) scatter，按 projection type 着色并报告相关性。
   - (b) 四个 depth bucket 上七类 projection 的平均 measured
     single-Linear \(\Delta\mathrm{NLL}\)。
   - 验证：PDF/PNG 均可生成，轴标签和图注不把 fitted proxy 写成 measured impact。
3. 将 PDF 接入 `references/wu.tex` 的 Figure 3，并收紧正文声明。
   - 验证：caption 明确模型、方法、数据集和 diagnostic 性质；正文仅声称图中实际展示
     的 local/depth/type 观察。
4. 进行离线校验并记录实现。
   - 验证：检查行数、数值有限性、输出文件存在；若本地 LaTeX 环境可用，检查 figure
     引用和资源路径。

