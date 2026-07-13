# Llama2 vLLM Prefill-Only Workload Search Plan

## Summary

- 目标是为 Llama2-7B + vLLM 找到最适合展示层异构压缩优势的 prefill-only 场景配置：`batch_size` 和 `input_seq`。
- 现有最终选择中的 `b8_in512_out1` 只比最佳统一压缩方法快 `1.020x`，展示性不足；需要针对纯 prefill 重新做 focused search。
- 不先重新压缩模型，优先复用已有 uniform/hetero checkpoint、质量代理、CUTLASS kernel predictor 和 vLLM benchmark 脚本，缩小到少量候选后再实测。

## Assumptions

- vLLM 当前 benchmark 仍用 `output_seq=1` 作为 prefill-only 代理；筛选时需要同时报告“纯 prefill 预测”和“vLLM out=1 实测/复测”两种口径。
- 统一压缩方法对比对象包括 `dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4`、`marlin_nvfp4`；层异构策略候选仍使用当前 vLLM fused Linear 约束。
- 精度先用已有 ARC-Challenge full `acc_norm` 和 quality proxy 过滤；最终候选必须用 vLLM + lm-eval 复测或复用已完成 full 结果。

## Key Steps

1. 汇总现有证据
   - 读取 broad-grid vLLM 统一方法速度、optimized/max-speed hetero retest、质量结果。
   - 输出当前已测 prefill-only 场景的 ranking，明确哪些场景已经不满足“明显好于所有统一方法”。
   - 验证：生成 CSV/Markdown，包含 hetero vs best uniform、acc_norm、acc loss。

2. 纯 prefill 预测筛选
   - 基于 `KernelLatencyPredictor` 只建模 prefill `M=batch*input_seq`，去掉 decode `M=batch` 的干扰。
   - 扫描候选 `batch_size` 和 `input_seq` 网格，计算每个 fused Linear 的最快方法分布、best mixed vs best single 预测收益。
   - 验证：候选表中能解释“为什么层间/Linear 间最佳方法不同”。

3. 生成 focused vLLM 复测清单
   - 选择预测收益最高且质量风险可接受的 3-5 个 `(batch,input_seq,output_seq=1)`。
   - 复用已有 policy solver/export/benchmark 脚本生成对应 hetero policy 和 vLLM benchmark 命令。
   - 验证：复测脚本可直接在 `conda activate cospaq` 后运行，输出 per-method median latency。

4. 最终推荐
   - 合并实测速度和质量，按 `hetero_vs_best_uniform`、`acc_norm_loss`、可解释性排序。
   - 给出首选配置、备选配置和不推荐配置。

## Success Criteria

- 至少给出一个 prefill-only 配置推荐，且有数据说明它是否真正明显优于所有统一压缩方法。
- 若现有数据没有满足条件的配置，明确指出原因，并给出下一批必须实测的最小候选集合。
- 所有新增分析结果和开发记录落盘，便于后续复现。
