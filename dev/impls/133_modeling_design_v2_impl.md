# 133 Modeling design v2 implementation

## 2026-07-23 - Detailed two-scenario design document

- 开发目的：形成能够完整解释当前建模、求解与真实验证流程的 v2 设计文档。
- 修改内容：核对 Llama3.1 prefill-only/prefill-decode 的最终实现，并新增
  `modeling_design_and_ablation_v2.md`。
- 影响文件：`dev/plans/133_modeling_design_v2_plan.md`、
  `dev/impls/133_modeling_design_v2_impl.md`、
  `artifacts/debug/064_llama31_modeling_ablation/report/modeling_design_and_ablation_v2.md`。
- 验证：关键脚本与冻结产物路径均存在，Markdown 的 110 个块级公式分隔符配对，
  `git diff --check` 通过；模型指标与 058/057/061 的 JSON 产物逐项核对。
- 后续注意：原始 `modeling_design_and_ablation.md` 和所有主实验结果保持不变。

## 2026-07-23 - Clarify Sparse BF16 traffic

- 开发目的：解释 Sparse BF16 Roofline 公式中 `1.777778` 的物理含义。
- 修改内容：将该常数展开为 \(16/9\)，补充 2:4 非零 BF16 values、位置 metadata 和
  \((9/8)NK\) 权重流量的逐步推导，并明确它不是拟合参数。
- 影响文件：
  `artifacts/debug/064_llama31_modeling_ablation/report/modeling_design_and_ablation_v2.md`。
- 后续注意：该公式是 packed representation 的解析近似，alignment、cache 和实际访问
  行为仍由硬件校准及 residual model 吸收。

## 2026-07-23 - Add metadata-aware Sparse NVFP4 traffic

- 开发目的：让论文中的 Sparse NVFP4 Roofline prior 显式反映 pairwise 4:8 metadata。
- 修改内容：补充 sparse FP4 values、位置 metadata、activation/weight scales 的逐项
  推导，将完整权重侧流量写为 \((11/32)NK\)，并记录 CUTLASS physical alignment。
- 影响文件：
  `artifacts/debug/064_llama31_modeling_ablation/report/modeling_design_and_ablation_v2.md`。
- 后续注意：冻结的 `sparse_nvfp4_model.py` 尚未显式加入 \(NK/16\) metadata 项；当前
  历史模型由独立硬件校准和 residual 部分吸收该误差。若论文声称使用修正后的逐项实现，
  应重新拟合 Sparse NVFP4 kernel model 并检查候选排序。
