# 009 Analysis Summary Plan

## 目标
- 分析 `artifacts/analysis` 下 NVFP4 microbench 结果。
- 按每个模型的 unique shape 聚合，避免重复 layer 干扰结论。
- 总结哪些 shape/配置加速或减速、效果如何以及可能原因。
- 如有必要在 `artifacts/analysis` 生成可视化和 `summary.md`。

## 步骤
1. 读取各模型 CSV，识别字段、状态、op 类型与可用样本。
2. 以模型内 unique shape 为核心聚合，分离模型级 forward、layer forward、quant/scale 等组件。
3. 生成必要的汇总表和图，辅助观察不同 shape 的加速/减速区间。
4. 写入 `artifacts/analysis/summary.md`，并在 `dev/impls/009_analysis_summary_impl.md` 记录实现。
