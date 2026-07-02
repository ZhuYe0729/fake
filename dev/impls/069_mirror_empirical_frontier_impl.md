# 069 MIRROR Empirical Frontier Implementation

## 2026-06-30 - Measured-result empirical frontier
- 开发目的：在 MIRROR 精度模型尚不稳定时，先基于已有完整实测点产出一套可交付 Pareto frontier。
- 修改内容：新增 `build_empirical_frontier_amp.py`，从 AMP-relative combined report 中计算 measured non-dominated frontier，输出候选表、推荐策略表、summary 和 PNG/PDF 图。
- 运行结果：生成 `report_empirical_frontier_amp/`；推荐 high-accuracy 点为 `batch_16_point_118`，balanced high-accuracy 点为 `batch_16_point_127`，max-speed usable 点为 `uniform_sparse_bf16`。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/build_empirical_frontier_amp.py`、`artifacts/debug/030_mirror_global_pareto/report_empirical_frontier_amp/`。
- 后续注意：该 frontier 是 measured candidates 上的临时实测前沿，不声称是全局最优；后续需要在精度模型修正后重新生成模型驱动候选。
