# Run015：Qwen3泛化、代表负载与建模开销

本目录保存对应人工reviewer的最终分析结果及冻结配置快照。完整代码、原始数据/日志和失败记录保留在[run015](../../../runs/015/README.md)，源代码没有搬迁。`code/reproduce_analysis.py`在同一仓库中调用run015源程序重建分析；需要原实验Python环境和原始归档。

[完整17行主表](../../reviewer2/qwen3_table2/README.md)使用统一RTX5090 GPU0、NUMA0、B8/S2048/O1或64、6GiB BF16 KV、decode Graph、三轮独立进程和BF16首尾锚点。新配置补完整质量，旧相同有效map按数据/权重/协议核验后继承。

代表负载为固定主配置迁移，不能当作逐负载重优化或逐负载质量保证。validation/test split不同，但test窗口沿用历史冻结集合；不宣称跨所有轮次全新未见测试集。epsilon是NLL代理预算，HAWQ是Hessian/ILP的PTQ移植。实测E2E、恢复完整cycle、准备和初始化分别报告；不以代理最优代替真实最优。
