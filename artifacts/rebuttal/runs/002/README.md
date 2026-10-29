# Qwen3-8B 8bit baseline 与 DASO 扩展实验

状态：已完成，完整性审计通过。计划见 [142 plan](../../../../dev/plans/142_qwen3_8bit_baselines_daso_plan.md)。

[完整报告](REPORT.md) · [最终 CSV](results/summary_final.csv) · [实测曲线](results/figures/quality_latency_zoom.png) · [审计](results/final_audit.json)

- 完成 240 个校准标签、47 个同卡速度配置（658 条记录）、57 个测试 NLL 点。
- 汇总 80 项完整 likelihood、57 份完整生成结果；其中新增 45 项 likelihood 和 36 份生成结果，其他为 001 冻结质量证据。新增生成覆盖 33,996 条样本。
- 新增 8bit 基线已完整评测。实际 prefill ΔNLL≤0.02 时 uniform SQ-INT8 达到 1.911×，当前扩展 DASO 为 1.225×；实际 decode ΔNLL≤0.01 时匹配五候选为 1.433×，扩展七候选为 1.366×。
- 原始 DP/质量模型保留；prefill 预算仍有违反，全局质量网格会过度向上取整 8bit 的小成本。详见报告和 [离散诊断](results/uniform_grid_audit.json)。

- 新增 SmoothQuant W8A8 INT8、原生 E4M3 W8A8 FP8，以及平滑 FP8 对照。
- 原五候选 + SmoothQuant INT8 / 原生 FP8，质量校准和策略搜索仅使用训练划分。
- 使用真实 `torch._int_mm` / vLLM SM120 CUTLASS FP8 GEMM；BF16 输出、注意力与 KV。
- 平滑匹配官方 Llama-like 范围（QKV、gate/up），alpha=0.5；阶段/模块内的逆平滑融合进动态 per-token 激活量化并计入实测成本。权重量化 per-output-channel。
- 同 001 的 B8/S2048/O1、O64、固定 3GiB BF16 KV 速度与 NLL 协议；生成任务固定 6GiB KV。
- 001 保持冻结；数据软链复用其已归档输入。临时可重建模型在 `/dev/shm/daso_rebuttal_002_20261025`。
