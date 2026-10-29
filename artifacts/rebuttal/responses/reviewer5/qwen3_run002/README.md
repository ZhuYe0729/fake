# gkP9：加入 INT8 / FP8 后的候选空间与实测前沿

结果来自完整性校验通过的 runs/002；旧 runs/001 证据保持不变。

- 三种新增 baseline：SmoothQuant INT8、原生 FP8、SmoothQuant-FP8 对照，均使用真实低精度 GEMM。
- 原五候选 + SmoothQuant INT8 / FP8 的 DASO，并有同质量拟合、同实测延迟表的五候选控制。
- 速度为同一物理 GPU3 的整模型测试；质量按独立测试 NLL 与完整任务报告。
- 平滑在候选内部融合进动态激活量化；保留 BF16 注意力与 KV。并非声称复现原论文的完整整型 attention pipeline。
- 质量预算实测与原网格离散诊断完整保留；本轮结果不能作为严格质量保证的证据。
- 与 001 HAWQ 的跨轮比较不能隔离分配器差异：INT8 内核实现、候选与延迟输入已变化。

[完整报告](../../../runs/002/REPORT.md)，[完整 CSV](summary_final.csv)，[质量—速度图](figures/quality_latency_zoom.png)。
代码、冻结数据和原始结果链接均指向本项目的归档实验目录。
