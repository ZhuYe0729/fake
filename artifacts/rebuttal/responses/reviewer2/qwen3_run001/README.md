# Exi5：HAWQ-v3 与 Qwen3 基线

包含七种 uniform、HAWQ-v3 分配器的 INT4 主对照及 FP4 控制、DASO 和阶段统一 hybrid。HAWQ 是公开公式/ILP 的 PTQ 移植，INT8 行不包含 SmoothQuant smoothing。

- [完整报告](../../../runs/001/REPORT.md)
- [最终结果表](results.csv)
- [策略与协议](configs/)
- [实验代码](code/)
- [冻结输入及逐模块误差](data/)
- [来源代码与论文参考](reference/)

代码、输入、配置和参考文件链接到同一份 run001 冻结来源；结果 CSV 和图表是最终快照。全量下游任务只对应预先选定代表点，其余扫描点评测真实速度和独立测试 NLL。
