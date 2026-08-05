# 135 Sparse NVFP4 metadata accounting plan

## 目标

统一建模文档中 sparse BF16 与 Sparse NVFP4 的解析 traffic 口径：论文推荐的
Sparse NVFP4 Roofline prior 显式计入结构化稀疏 metadata，同时如实记录当前冻结实现
仍使用省略该项的简化式。

## 实施

1. 在 Sparse NVFP4 W4A4 小节加入 value、metadata、activation/weight scale 和 output 的完整 traffic 分解。
2. 给出与 CUTLASS SM120 对齐 layout 一致的 metadata physical-buffer 公式，并说明 Llama 主 shape 上的简化关系。
3. 说明当前 `sparse_nvfp4_model.py` 的冻结 prior 未显式计入该项；benchmark/residual 已部分吸收其影响，若将新 prior 用于主预测器需重新拟合并验证。

## 验证

- Sparse NVFP4 解析式显式包含 \(NK/16\) metadata，且与 sparse BF16 的 metadata-aware 表述一致。
- 文档不将未重训的 metadata-aware prior 误称为当前冻结结果的逐项实现。
