# 133 Modeling design v2 plan

## 目标

在不修改现有实验结果和实现代码的前提下，新增一份完整的建模与端到端方法说明，
明确区分 Llama3.1-8B-Instruct 的 prefill-only 与 prefill-decode 场景。文档应从
canonical 压缩语义、质量与速度特征、拟合、离散求解、真实 vLLM closure 一直解释到
最终论文产物，并给出与实际代码一致的公式、数据结构、伪代码和路径索引。

## 实施

1. 核对 057/058/061/064/068 中冻结的质量标签、局部误差、速度模型、E2E 校准、
   动作支持、动态规划求解和真实测试口径。
2. 在 `artifacts/debug/064_llama31_modeling_ablation/report/` 新增
   `modeling_design_and_ablation_v2.md`，保留原文档不变。
3. 分别写出两个场景的目标定义、质量公式、速度公式、求解公式、关键代码和 closure。
4. 统一主实验采用 PAVA 的表述，并明确预测筛选与论文实测结果的边界。

## 验证

- 文档中的常数、张量维度、训练/holdout 数量和任务协议与实际产物一致。
- 所有关键实现路径存在。
- Markdown 公式、代码块和章节层次可直接阅读。
- 不覆盖 057/058/061/064/068 的已有数据和主结果。
