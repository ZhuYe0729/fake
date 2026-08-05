## 2026-07-23 - 补充单模块真实 NLL baseline 说明
- 开发目的：澄清策略级质量模型、模块可加代理与约束优化之间的关系。
- 修改内容：在建模文档 Appendix 中加入单模块真实 NLL 可加 baseline 的定义、适用限制和对比协议。
- 影响文件：`artifacts/debug/064_llama31_modeling_ablation/report/modeling_design_and_ablation_v2.md`
- 后续注意：如实际运行该 baseline，应补充同一 holdout 上的指标和等预算对比结果。

## 2026-07-23 - 记录 PAVA 与固定 offset 的校准诊断
- 开发目的：为 E2E 采用 PAVA 而非 \(g(x)=x+C\) 提供可验证的实证依据。
- 修改内容：在 `TODO_tmp.md` 记录残差趋势检查，以及 raw、fixed-offset、affine、PAVA 的同 holdout 对比协议。
- 影响文件：`TODO_tmp.md`
- 后续注意：仅当 PAVA 在 holdout 上有稳定增益时，将其作为主校准器的经验论证。

## 2026-07-23 - 修正 TODO 行内公式渲染
- 开发目的：使 PAVA 诊断记录可在 Markdown 渲染器中稳定显示。
- 修改内容：将 `TODO_tmp.md` 中的行内 LaTex 从 `\\(...\\)` 统一为 `$...$`。
- 影响文件：`TODO_tmp.md`
- 后续注意：后续 TODO 中的行内公式沿用 `$...$` 写法。

## 2026-07-23 - 区分 PAVA 动机实验与完整验证
- 开发目的：优先以低成本实验说明固定 offset 假设的局限，保留完整泛化验证作为后续工作。
- 修改内容：将 TODO 划分为现有 calibration 数据上的 motivation 图，以及 train/holdout 的完整比较。
- 影响文件：`TODO_tmp.md`
- 后续注意：motivation 图不能替代 holdout 指标，也不应作为泛化结论报告。
