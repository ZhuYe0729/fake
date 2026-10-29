# llama2_7b_chat统一阶段控制

等待037基线重放核查后，补plain INT8/FP8 prefill＋W4A16 decode两固定控制，与原SQ对应点及BF16/INT8/SQ/FP8比较。2新NLL、6新完整生成、7map统一三轮速度；不标为DASO求解器结果。原数据与框架不改。

[计划](../../../../dev/plans/164_llama_uniform_phase_controls_plan.md) · [固定配置](configs/sweep_design.json) · [状态](results/sweep_worker.json)
