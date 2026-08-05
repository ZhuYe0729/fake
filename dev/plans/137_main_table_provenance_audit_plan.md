# Main Table Provenance Audit Plan

## 目标

审计 `artifacts/debug/071_my_emergency/main_table.tex` 中 RTX 5090 主结果表的全部
质量与速度数据，逐项追溯到非 summary 的原始测量文件，并判断是否存在旧协议、不同
配置或跨实验不一致的数据混用。

## 步骤与验证

1. 建立表格行与实际 policy 的映射。
   - 验证：每个 prefill-only 和 prefill--decode 半行都有唯一 policy ID、policy JSON
     与实验目录。
2. 追溯原始质量和速度测量。
   - 验证：质量值能从 lm-eval 或生成任务原始 `result.json` 复算；速度值能从原始
     runs JSON/CSV 的 BF16 anchor 与 policy latency 复算。
3. 核对实验协议与配置。
   - 验证：记录模型、GPU、batch size、输入/输出长度、runtime、KV cache、模板、
     few-shot、metric 和 policy hash；明确所有不一致或缺失项。
4. 输出详细 Markdown 报告。
   - 验证：报告覆盖表中每一行，区分原始测量、派生值和仅有汇总来源的值，并给出
     “可用 / 有条件可用 / 应重测”结论。

## 成功标准

- 不使用 summary Markdown 作为最终数据证据。
- 每个展示值都有原始文件路径或明确记录“未找到原始文件”。
- 对 `point_007` 是否属于当前、同配置且非过时结果给出证据化结论。
