# SQ-INT8 prefill / W4A16 decode 快速对照

用户确认固定SQ-INT8 prefill，只替换decode。独立027不改旧策略或主表；四配置为新组合、全SQ-INT8、普通INT8、旧DASO-7 ε=.01。每任务按固定seed从原冻结输入抽128样本，全部新跑同样greedy/256上限/Graph协议。

生成子集为探索性试验，不视作新独立测试语料或完整任务结果。保持prefill相同的因果对照是新组合与全SQ-INT8；旧DASO参考的prefill为NVFP4。速度/NLL优先复用019同map三轮结果，普通INT8旧轮速度另标来源。使用共享测量锁和空闲卡；正在运行的独占测速优先，不中断既有任务。

为避免128条子集波动遮蔽小幅差距，在任何027 GPU输出前固定补齐唯一新组合的完整三项生成（1000/1500/333），复用同一checkpoint；其他三配置仍新跑128条，对照完整分数引用原归档并标来源。full/下保存完整输入链接与独立输出，不覆盖pilot。

CNN pilot对历史输出未全匹配（新组合112/128、SQ对照88/128；抽样也改变batch分组），因此补全SQ-INT8对照完整三任务，新组合与SQ对照全量均为027 Graph新测。额外仅用1张空闲卡并遵循同一共享锁。普通INT8/旧DASO的完整值仍明确列历史来源；四配置128条均新测。

最终状态：已完成12项128条pilot与6项完整生成。[完整报告](REPORT.md) · [完整分数CSV](results/full_generation_comparison.csv) · [交付检查](results/delivery_checks.json)。旧主表未替换，固定组合的NLL预算与cycle开销均单列。
