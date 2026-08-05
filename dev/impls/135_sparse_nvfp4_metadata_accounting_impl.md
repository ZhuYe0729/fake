## 2026-07-24 - 统一 Sparse NVFP4 metadata 解析口径
- 开发目的：使论文推荐 Roofline prior 对 sparse BF16 与 Sparse NVFP4 一致计入稀疏 metadata。
- 修改内容：在建模文档中补充 Sparse NVFP4 的 metadata-aware traffic 公式、对齐后的 physical-buffer 形式及冻结实现差异说明。
- 影响文件：`artifacts/debug/064_llama31_modeling_ablation/report/modeling_design_and_ablation_v2.md`
- 后续注意：若同步修改生产 predictor，必须重新拟合 Sparse NVFP4 residual 并复核 unseen-shape 误差和 solver 排序。
