# 2026-10-27夜间补充实验

用户授权约九小时持续推进，窗口15:53 UTC至次日00:53 UTC。已启动的必要实验在窗口结束时仍需核实完成，不把排队当作完成。源码、日志、CSV、文档和失败记录均保留；只清理本任务可重建权重二进制。

[全部实验进展](TRACKER.md) · [状态JSON](status.json) · [计划161](../../../dev/plans/161_rebuttal_nine_hour_followthrough_plan.md) · [原审稿实验清单](../EXPERIMENT_CHECKLIST.md)

| Run | 模型与用途 | 对应reviewer | 测试范围 | 交付位置/状态入口 |
| --- | --- | --- | --- | --- |
| 034 | Llama3.1六预算/权重点 | R1/R2/R3/R5 | 完整八候选独立阶段求解、六新NLL/18完整GEN、11map三轮速度 | [已完成报告](../runs/034/SWEEP_REPORT.md) |
| 035 | Llama2六预算/权重点 | R1/R2/R3/R5 | 同上；GPU0被占用后保留16份部分速度，GPU1重新完整三轮 | [已完成报告](../runs/035/SWEEP_REPORT.md) |
| 036 | Llama3.1中间权重补充 | R1/R2/R3 | wP16/q=.03，完整质量及GPU4同卡三轮速度；资源干扰重跑均保留 | [已完成报告](../runs/036/SWEEP_REPORT.md) |
| 037 | 原基线数值复验 | R2/R3 | 两Llama×三基线，原64窗口NLL通过；2304生成序列全部逐token相同 | [已完成复核](../responses/reviewer2/llama_baseline_replay_run037/README.md) |
| 038 | Llama3.1固定阶段控制 | R2/R3 | plain INT8或FP8 prefill＋W4A16 decode，两新map完整质量，七map三轮速度 | [状态](../runs/038/results/sweep_worker.json) |
| 039 | Llama2固定阶段控制 | R2/R3 | 与038相同设计 | [状态](../runs/039/results/sweep_worker.json) |
| 040 | Qwen新点跨六负载迁移 | R2/R3/R4 | 两新DASO点及七强参考、九map×六case，逐case64窗口质量和三轮速度 | [状态](../runs/040/results/worker.json) |
| 041 | Llama3.1同候选与反约束 | R2/R5 | 原18个DASO/Hessian点，补齐r=.95/.80/.65实际速度、原上限超限、配对NLL误差 | [状态](../runs/041/results/worker.json) |
| 042 | Llama3.1 SQ-FP8 | R2/R3 | 固定alpha=.5/scales，完整MC/GEN、两场景NLL，五map两场景统一速度 | [状态](../runs/042/results/worker.json) |
| 043 | Llama2 SQ-FP8 | R2/R3 | 与042相同设计，保留原B4生成、B8 MC/速度及10GiB评估KV | [状态](../runs/043/results/worker.json) |
| 044 | Qwen新窗口预算复核 | R1/R3 | 固定12map，64 validation＋64 confirmation，先冻结选点再确认 | [状态](../runs/044/results/worker.json) |
| 045 | Llama3.1新窗口预算复核 | R1/R3 | 固定12map，与044相同隔离设计 | [状态](../runs/045/results/worker.json) |
| 046 | Llama2新窗口预算复核 | R1/R3 | 固定11map，与044相同隔离设计 | [状态](../runs/046/results/worker.json) |
| 047 | Qwen新点真实mixed调度 | R2/R1/R5 | 固定九map/两chunk1024 burst，64窗口实际路由NLL及三轮速度/显存；等待048核心重复完成 | [状态](../runs/047/results/worker.json) |
| 048 | Qwen十配置统一完整重复 | R2/R3 | 当前同一Graph协议，040九map加SQ-FP8，30项完整GEN/33正式进程；等待044–046，优先于047 | [状态](../runs/048/results/worker.json) |

阶段控制不是DASO自动选择。非单位prefill权重是偏好预算，不是重新标定后的物理ΔNLL。新选点均来自已查看benchmark上的探索，真实NLL未通过名义q时明确记录，不因生成平均分较高就标为预算保证。

CPU统计：三个模型的5000次配对描述性bootstrap均已完成：[Qwen](../responses/reviewer2/quality_uncertainty_night161/qwen3/README.md)、[Llama3.1](../responses/reviewer2/quality_uncertainty_night161/llama31/README.md)、[Llama2](../responses/reviewer2/quality_uncertainty_night161/llama2/README.md)。这些区间未校正选点/多重比较，不能称独立盲测。Llama2 wP32的宏平均差相对INT8区间包含0，不能据小幅点估计宣称精度显著提升。

034/035/036已审计并发布追加主表：[Llama3.1共27行](../responses/reviewer2/llama31_8b_instruct_table2_sweep/README.md)、[Llama2共26行](../responses/reviewer2/llama2_7b_chat_table2_sweep/README.md)，原15行保持不动。后续阶段控制与SQ-FP8完成后另建30/29行汇总，保留全部有利/不利点和各自campaign来源。

新排队：044/045/046为Qwen/Llama3.1/Llama2的新窗口预算复核，分别固定12/12/11个map；等待043结束，64validation与64confirmation源位置及16gram不交叉。详见[计划168](../../../dev/plans/168_fixed_pool_fresh_quality_audit_plan.md)。

[新点准备/恢复开销](../responses/reviewer3/new_point_overheads_night161/README.md)已完成五campaign的最终快照，含21个求解标签、42个实际速度map及成功/失败叶子成本；旧中间快照保留。

延迟校验补充：[计划171](../../../dev/plans/171_preregistered_latency_gate_plan.md)的规则/执行源码已在041首个速度输出前冻结。复用round0校验、round1/2检查，E2E/cycle分别处理，等待041完成后自动分析；不增加GPU测量，不覆盖原绝对上限或原始违约。

统一重复补充：[计划172](../../../dev/plans/172_qwen_full_generation_repeat_plan.md)。002原生成是eager，028/032后续为decode Graph/arena，INT8分数有差异。048对全部十map统一当前协议，保留旧24行，完成后另建34行表；不预设差异原因或DASO必胜。047 mixed在048后开展。048完成后，新增参考bootstrap另写quality_uncertainty_all_baselines_night161，原区间不覆盖。
