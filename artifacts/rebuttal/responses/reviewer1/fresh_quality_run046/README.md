# llama2_7b_chat：固定池的新窗口质量预算复核（R1 AK6a / R5 gkP9）

固定11套既有map，未重拟合质量代理或调q/wP。以新validation32/64窗口的mean+2SE+.005保护值筛选，在四个实际epsilon下选冻结Linear LUT成本最低的incumbent；BF16保护值为0。仅有限池筛选，不称完整空间重新优化。

数据来自公开WikiText train，排除旧实验校准128个P+D/100个prefill样本和完整旧validation/test分词流中共享的任何连续16token片段。新validation64与confirmation64互不重叠，数据和种子先于GPU输出冻结。不是外部未见语料，不是新的下游任务盲测；NLL窗口SE和保护值均为经验诊断，不是泛化定理。

选择文件在confirmation前冻结。八个预算标签实际对应5个选择map，确认并集另含BF16和三个8bit参考，共6个map；重复选点不能当作独立成功次数。B8/S2048/O64、KV10GiB、原native decode Graph与数值代码，重导出权重完整SHA须与validation一致。

| Validation N | epsilon | 配置 | Validation保护值 | 新确认ΔNLL | ±2SE | 确认保护值 | 均值超限 | 保护值超限 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 32 | 0.010 | uniform_dense_bf16 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | False | False |
| 32 | 0.020 | uniform_int8 | 0.01745 | 0.00428 | 0.00720 | 0.01648 | False | False |
| 32 | 0.040 | uniform_sq_int8 | 0.03348 | 0.02317 | 0.01031 | 0.03848 | False | False |
| 32 | 0.070 | w4_q0.03 | 0.06520 | 0.05251 | 0.01765 | 0.07516 | False | True |
| 64 | 0.010 | uniform_dense_bf16 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | False | False |
| 64 | 0.020 | uniform_int8 | 0.01012 | 0.00428 | 0.00720 | 0.01648 | False | False |
| 64 | 0.040 | w32_q0.03 | 0.03867 | 0.02353 | 0.01161 | 0.04014 | False | True |
| 64 | 0.070 | w4_q0.03 | 0.06254 | 0.05251 | 0.01765 | 0.07516 | False | True |

原q/wP标签与配置来源见design.json/policy origins；本轮epsilon是validation实际筛选预算，不能把原偏好q和它混用。本轮没有新增GEN或速度；相同map的旧下游分数/速度见各origin run的完整报告与来源审计，不能称新独立结果。

[全部选择及超限](quality_budget_audit.csv) · [唯一确认map](unique_confirmation.csv) · [全部validation候选](validation_candidates.csv) · [冻结选择](selection.json) · [固定池与来源](design.json) · [数据排除](data_purge.json) · [审计](provenance.json)
