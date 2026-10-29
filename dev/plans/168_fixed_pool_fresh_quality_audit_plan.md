# 新选点固定池的独立窗口质量预算复核（R1 AK6a / R5 gkP9）

2026-10-27，161九小时后续；run044 Qwen、045 Llama3.1、046 Llama2。全部前序任务完成后开展，不抢已排正式测速。

1. 固定032/033、034/036、035已经冻结并完成评测的全部探索map，加五个原八候选空间内参考：BF16、INT8、SQ-INT8、FP8、SQ-prefill/W4-decode。去重并保留标签来源，不重新训练质量代理或调q/wP；候选池是有限incumbent池，不声称重新搜索完整空间。
2. 各模型使用已有WikiText train分词流，排除与实验校准128个P+D及100个prefill样本共享任何连续16token片段的2112-token窗口。旧实验validation/test来源为其他语料split。64-bit hash碰撞只会保守多排除，不会漏掉相同16token串。余下对齐窗口用固定seed抽取源位置不重叠且两fold不共享16token片段的新validation64和confirmation64；数据/SHA在本轮GPU输出前冻结。
3. 语料是公开已使用WikiText train，不称外部未见语料，也不把这轮NLL复核当下游任务独立盲测。Llama2旧calibration的全行与当前train分词流不完全一致，因此统一采用上述16token排除标准，不虚称所有旧样本均已定位为完整区间。
4. 所有固定候选在新validation64上实测P+D NLL，分别用前32和全部64窗口计算paired mean+2SE+.005保护值（BF16=0）。在epsilon=.01/.02/.04/.07下，从通过保护值的固定池中选冻结实测LUT Linear成本最低的map。记录池规模、重复选点及BF16回退。
5. 选择完成后写入带SHA的selection文件，之后只对选择并集、BF16与三个8bit参考运行新confirmation64；不根据其结果重新选点。报告真实均值/保护值超限、配对SE、相同map复用的既有下游分数/速度来源，不能声称新增了完整任务或速度测量。
6. 同原B8/S2048/O64/6或10GiB KV/native decode Graph、数值代码/源模型/scales保持。串行导出锁及空闲卡质量并行；每次验证后可清自己可重建权重，若确认阶段重导出则核对完整model.safetensors SHA等于validation表示。所有其他文件保留。

验证：数据窗口无交叉、无校准16gram hash交集；全部map与输入/源审计冻结；validation与confirmation严格按流程隔离，selection先于confirmation且未修改；每个实际配置64窗口，不把重复预算标签当独立样本；全部失败/不利结果保留。经验保护值不构成数学泛化保证。
