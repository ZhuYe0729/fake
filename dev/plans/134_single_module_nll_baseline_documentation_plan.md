# 134 Single-module NLL baseline documentation plan

## 目标

在建模设计文档的 Appendix 中补充质量可加代理的解释，并定义单模块真实 NLL 可加
baseline，避免将模块动作代价误读为可直接观测的因果效应。

## 实施

1. 说明质量模型只由策略级真实 \(\Delta\mathrm{NLL}\) 监督，\(q_{i,a}\) 是用于求解的可加代理。
2. 给出“仅压缩一个模块”的真实 NLL baseline 及其对 mixed policy 的加和预测式。
3. 说明该 baseline 应在相同 mixed-policy holdout 上比较，并注明真实评测预算差异。

## 验证

- Appendix 的公式与第 3、9 节的质量模型和 knapsack 形式一致。
- 不将尚未运行的 baseline 写成已有实验结果。
