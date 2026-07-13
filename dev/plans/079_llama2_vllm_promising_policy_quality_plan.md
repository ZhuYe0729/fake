# Llama2 vLLM Promising Policy Quality Plan

## Summary

- 对 `promising_policy_retest` 中 8 个 unique optimized hetero policy 做 full ARC-Challenge quality eval。
- single 方法精度复用 018 full ARC-C 已有结果，不重复测试。
- 将策略精度、best-single speedup 对比和策略明细补入 promising policy 的最终宽表文档。

## Key Steps

1. 扩展 quality eval：
   - 复用已有 `lm_eval` + vLLM 评测方式。
   - 输入为已经按上层压缩算法导出的 vLLM checkpoint。
   - 每个 unique policy 只评测一次，再按场景映射。

2. 汇总精度：
   - 输出每个 policy 的 `arc_acc`、`arc_acc_norm`、`sample_len`。
   - single 方法精度从 `018_llama2_prefill_global_pareto/report/final_full_arc_c_report.csv` 读取。

3. 更新宽表：
   - 增加 `optimized_vs_best_single_speedup`，括号标注 best single 方法。
   - 增加 `optimized_arc_acc_norm`。
   - 在文档底部增加 single 方法精度小表。
   - 增加 policy 明细表，展示每个 policy 的方法计数、质量成本和服务的场景。

## Assumptions

- 精度指标继续使用 018 中的 full ARC-Challenge 0-shot `acc_norm` 作为主指标。
- speedup 仍统一用同场景 dense bf16 vLLM median latency 作为基准。
- 8 个 optimized policy checkpoint 已经由严格压缩导出流程生成，本次精度测试直接加载这些导出模型，不重新构造压缩。
