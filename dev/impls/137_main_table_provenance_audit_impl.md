## 2026-07-28 - 主结果表原始数据溯源审计

- 开发目的：确认 `main_table.tex` 的质量与速度结果是否来自当前、同配置实验，并为每行提供非 summary 的原始数据路径。
- 修改内容：逐项映射四条实验轨道和 policy，复算质量指标与 BF16 speedup，核对 warmed protocol、workload、prompt template 和后续实验；输出详细 provenance 报告。
- 影响文件：`artifacts/debug/071_my_emergency/main_table_provenance_report.md`
- 审计结论：Llama-3.1 Balanced 的 `point_007` MMLU 66.50 可由 full 0-shot 原始 lm-eval JSON 验证；发现 Llama-2 三组生成 baseline 缺逐样本原始文件、Llama-2 Sparse NVFP4 decode 使用 Dense NVFP4 legal projection，以及 Llama-3.1 generation 使用 legacy/common PMPD prompt。
- 后续注意：论文提交前应补跑缺失 baseline、披露 prompt protocol，并在表注中修正 legal-projection 标签。
