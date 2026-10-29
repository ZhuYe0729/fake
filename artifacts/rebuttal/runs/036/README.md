# Llama-3.1中间prefill权重单点补充

036只测wP16/q=.03，128模块×64动作自由搜索；原034/035六点不改。本补充在已看到034前两个原权重生成结果后，依据CPU分配设计：prefill无FP4，仅5BF16（wP32为13BF16），其余91SQ/24INT8/8FP8，decode109W4/19INT8。

1个新NLL、3个新完整生成、5份同协议参考，三轮统一单卡速度；质量遵守原两组测速优先，正式速度等待034/035审计完成。全部点/失败保留，不称新盲测或通用最优。

[冻结配置](configs/sweep_points.json) · [状态](results/sweep_worker.json) · [计划](../../../../dev/plans/160_llama_budget_prefill_weight_sweep_plan.md)
