## 2026-07-13 - Pareto design documentation
- 开发目的：为论文 Design 部分沉淀当前 Llama2-7B-chat phase-heterogeneous Pareto 方法的准确、可复现描述。
- 修改内容：新增详细中文设计文档，覆盖 fused action space、WikiText local+global calibrated NLL proxy、kernel latency surrogate、policy-level isotonic E2E calibration、显存可行性、multiple-choice knapsack DP、实测闭环、论文表述边界和产物索引。
- 影响文件：`dev/094_llama2_prefill_decode_pareto_design.md`、`dev/plans/094_llama2_pareto_design_document_plan.md`。
- 后续注意：文档明确当前 034 DP 尚未将 035 isotonic calibration 回灌进 solve loop；论文最终版本实现该步骤后，应同步更新“当前实现状态”和检查清单。

## 2026-07-13 - Roofline speed-model expansion
- 开发目的：补全速度模型的底层物理结构，避免将 calibrated roofline 错写为纯黑盒 latency regression。
- 修改内容：核对 kernel modeling 实现后，新增各 kernel 的 effective work/traffic、实测 99 分位有效吞吐/带宽与 launch floor 标定、shape floor、最近邻+ridget residual correction、phase-policy aggregation 和 policy-level isotonic E2E calibration 的公式与边界。
- 影响文件：`dev/094_llama2_prefill_decode_pareto_design.md`。
- 后续注意：论文应称 kernel-specific profile-calibrated roofline model，而非只按 GPU datasheet 的纯解析 roofline。
