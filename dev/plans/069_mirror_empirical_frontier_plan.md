# 069 MIRROR Empirical Frontier Plan

## 背景

当前 MIRROR 精度模型对 sparse_bf16 / mixed backend 策略的 NLL 排序仍不稳定，短期内继续依赖模型预测来选 Pareto 点风险较高。为了先形成一套可交付结果，先基于已经完成的端到端实测结果构造 empirical frontier。

## 假设

- 现有 `report_keyfix_genimage_theoretical_clean_frontier_bf16_relative/combined_report_bf16_relative.csv` 已包含完整质量和速度实测点，并且 x 轴已修正为 dense-default + AMP baseline。
- 先接受“实测点集合较少”的限制：该 frontier 是 measured candidates 的非支配前沿，不声称覆盖全局最优。
- 后续精度模型修正后，可以用模型驱动候选替换这版 empirical frontier。

## 步骤

1. 从已有完整实测点中计算非支配点。
   - verify: 输出 frontier CSV，所有入选点在 `speedup_vs_dense_default_use_amp` 和 `bal_acc` 上不被其他实测点同时支配。
2. 生成临时可交付报告。
   - verify: 输出策略摘要、候选表、PNG/PDF 图。
3. 标记建议策略。
   - verify: 明确 high-accuracy、balanced、max-speed 三类可用策略及其 policy 路径。

## 产物

- `artifacts/debug/030_mirror_global_pareto/report_empirical_frontier_amp/`
