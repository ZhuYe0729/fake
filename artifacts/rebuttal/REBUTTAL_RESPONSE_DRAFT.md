# Rebuttal逐reviewer回复草稿

更新：2026-10-29。以下英文段落可直接编辑；每段后的中文说明及本地链接供内部核对，不属于需要提交的正文。未假定论文正文已修改，也未替用户提交回复。

共用口径：实际质量预算ε约束相对BF16的平均ΔNLL（nats/token）；探索配置的q/wP不是实际ε保证。速度均相对对应campaign的BF16；完整cycle另计下一批prefill前的恢复。主表的生成任务分项独立报告。

## R1 / AK6a：预算满足率与双表示开销

> In an expanded Qwen3-8B prefill–decode audit, we evaluated all 18 predeclared budget–workload settings: nine workloads at ε=0.01 and 0.07, using the original DP solver with the seven-candidate extension and 100 fixed WikiText test windows per setting. Measured mean ΔNLL satisfied the budget in 17/18 settings (94.4%): 8/9 at ε=0.01 and 9/9 at ε=0.07. The only violation occurred at B=8, input/output lengths 512/64, and ε=0.01; the measured mean was 0.011959, exceeding the budget by 0.001959 nats/token. These settings reuse 16 distinct policies and shared text windows, and are not 18 independent trials.
>
> The quality model remained frozen across workloads, and test results were not used to select policies. The paper reports held-out ΔNLL MAE of 0.0413 and Spearman correlation of 0.963 on Llama-3.1, supporting useful overall prediction and ranking rather than exact budget certification. Under tight budgets, absolute surrogate residuals and finite-sample uncertainty can be comparable to ε. The observed excess is small in absolute NLL, but this audit does not isolate its cause or establish downstream equivalence. Additional measured validation provides a further check when strict budget satisfaction is required.
>
> On the same arena/Graph path as the Qwen main table, with a fixed 6 GiB KV cache, the two reported DASO policies use 15.548/16.269 GiB peak Torch-allocated GPU memory, versus 23.041 GiB for BF16. They additionally retain 8.236/8.757 GiB of CPU phase-weight cache. This persistent CPU requirement does not directly occupy GPU memory; actual phase-transfer and scheduling costs are reflected in the corresponding E2E and restoration-inclusive cycle measurements. Identical representations are shared; different phase representations reuse per-layer GPU buffers rather than keeping two complete GPU models resident.
>
> Dedicated-stream prefetch overlaps representation preparation with subsequent prefill computation. The paper's representative Llama-3.1 profile reports 354.94 ms cumulative copy work and 1.92 ms exposed copy waiting, with a separate runtime ablation reducing median E2E latency by 6.4%. These are diagnostic and ablation results, not a universal zero-overhead claim. In the Qwen main-table experiment, including the 90/102 ms restoration before the next batch still yields 1.844×/1.842× BF16 cycle speedups. Peak memory can nevertheless limit KV capacity and concurrency. Our current objective optimizes latency under a quality constraint and does not explicitly constrain peak GPU memory. Future work will explore explicit memory budgets, finer-grained bounded prefetch, and buffer-lifetime management to improve the memory–latency trade-off.

对应[P+D文字回复与口径](responses/reviewer1/quality_budget_text_reply/README.md)、[完整18设定及逐窗重算](responses/reviewer1/pd_quality_budget_expanded/README.md)与[运行开销](REBUTTAL_SELECTED_EVIDENCE.md#runtime-cost)。

措辞要点：18=9负载×2预算，16套唯一配置，文本复用；不是18种预算或18次独立随机实验。唯一超限0.001959只说明本预设网格的结果，不证明一般违约概率、样本主因、策略选择无影响或下游等效。该组是Run003原DP七候选扩展的跨负载审计，与最近053速度及旧Run005四预算/附加MILP分别报告，不能合并频率。原24标签validation增强版继续完整保留。

## R2 / Exi5：强基线、负载覆盖与mixed forward

> We added Qwen3-8B beyond the Llama family and expanded the baselines to INT8, SmoothQuant INT8, FP8, SmoothQuant FP8, a HAWQ-V3-style hardware-latency/Hessian ILP adaptation for LLM PTQ, and simple uniform phase combinations. The Llama evaluations were also extended with these baseline families.
>
> In the unified Qwen3 P+D repeat, two explored DASO policies achieve 2.017×/2.032× BF16 E2E speed, versus 1.665× for INT8 and 1.934× for the SQ-INT8-prefill/W4A16-decode control. Including restoration, they remain 1.108×/1.106× faster than INT8. We report the three generation metrics separately: the first DASO policy obtains 22.707/27.335/39.995 versus BF16's 22.824/26.957/40.132. These are measured quality–speed trade-offs; actual NLL-budget checks are reported separately. Both frozen policies also have faster median E2E and full-cycle times than INT8 across six batch/input/output configurations, without per-workload retuning.
>
> At the reported balanced operating points, DASO reduces prefill TTFT relative to the HAWQ-V3 adaptation by 27.7%–32.4% across (batch, input length) = (8, 2048), (4, 2048), and (8, 4096). Within each scenario, these representative policies are frozen across workloads, with three complete matched rounds on one GPU. We additionally report both E2E latency and restoration-inclusive cycle time for four P+D workloads, together with repeat ranges and full original-protocol task scores.

新增多配置来源：[053完整双场景表](responses/reviewer2/qwen_night_cohorts_run053/README.md)。以上27.7%–32.4%只描述所报告的均衡点Prefill比较；新表采用统一扩展引擎容量，与048原主表分开引用。P+D的E2E与cycle存在取舍，†点有超过5%的轮间变化，不能将该段改成全场景最优声明。

> Mixed prefill/decode packing does not require all token rows to use the same Linear weights. We use scheduler positions and original prompt lengths to gather prefill and decode rows, execute their corresponding phase kernels, and scatter outputs back to the original order before subsequent operations. Attention retains the original request boundaries and KV metadata; a one-token prompt chunk is still classified as prefill. Our vLLM V1 implementation, with chunked-prefill token budget 1024, records 15/10 genuinely mixed forwards per timed burst on the standard/heterogeneous workloads. Same-row-group kernel-reference checks and BF16 no-op NLL consistency pass. Across three rounds, the balanced/speed-oriented policies achieve 1.348×/1.225× BF16 E2E speedups on the standard workload and 1.013×/1.210× on the heterogeneous long-decode workload. This extension uses dual-resident eager execution and all-at-t0 closed-loop bursts, separately from the main-table single-arena Graph path and from a production arrival-stream benchmark.
>
> PD-disaggregated serving provides another architectural mapping: prefill and decode workers can hold their respective phase configurations while exchanging compatible KV caches. [vLLM documents two-instance KV-transfer support](https://docs.vllm.ai/en/v0.11.1/features/disagg_prefill/). This is a compatibility argument, not a measured distributed-system speedup: actual KV handoff, cache layout/position consistency, and ownership of the first generated token still require a two-worker test.

对应[模型主表](REBUTTAL_SELECTED_EVIDENCE.md#model-results)、[不同负载](REBUTTAL_SELECTED_EVIDENCE.md#workloads)和[混合调度](REBUTTAL_SELECTED_EVIDENCE.md#mixed)；同轮主比较来自[048](responses/reviewer2/qwen_full_repeat_run048/README.md)，负载来自[040完整九方法质量–速度对照](responses/reviewer2/qwen_workload_comparison/README.md)，mixed来自[047](responses/reviewer2/qwen_updated_mixed_run047/README.md)。[mixed/PD兼容性中文回复、来源复核与补测建议](responses/reviewer2/mixed_pd_compatibility/README.md)。[六负载质量–速度可导出图](responses/reviewer2/qwen_workload_comparison/quality_speed.pdf)。负载表精度为实测ΔNLL，不是逐负载生成任务Avg。

若需要进一步回应“收益是否只来自扩展候选”，可追加：

> In a controlled Llama-3.1 comparison with identical candidates, latency cap, and MILP solver, replacing the signed-Hessian quality proxy with DASO gives measured ΔNLL 0.00345 versus 0.00794 at similar E2E latency (1471.71 versus 1476.07 ms). This isolates the proxy comparison from candidate-space and solver differences.

该追加例对应[041的r=.80 phase-pair对照](responses/reviewer2/llama31_matched_allocator_run041/README.md)，仅针对signed-Hessian适配，不泛化为击败所有Hessian变体；HAWQ的延迟目标与DASO实际质量预算不是同一约束。

## R3 / LrT2：预测准确性、非Linear与硬件校准成本

> We added Qwen3-8B to extend the evaluation beyond Llama and now report two distinct latency validations. For the original hybrid estimator, held-out shape and N/K-group MAPE is 4.18%–11.83% across five backends. Separately, the measured-LUT deployment extension has policy-level E2E MAPE of 1.00%–1.71% across Qwen3 and both Llama models. The latter validates known profiled shapes and is not presented as unseen-shape prediction.
>
> Measured backend costs include internal activation quantization, scaling, and metadata processing. Non-Linear work, including attention, KV operations, normalization, activations, and engine overhead, is approximated using a BF16 residual fixed before allocation; it is not assumed to be zero. Restoration is reported separately through full-cycle latency.
>
> Hardware-specific profiling is required. The additional 80-row Linear profiles for Llama-3.1/Llama-2 took 38.19/28.62 seconds; their new 8-bit feature collection took 98.34/90.77 seconds. These are incremental function-level costs using existing complete-policy labels. We account for label collection, export, and validation separately, rather than equating the short fitting or solving time with the entire preparation cost.

对应[建模与求解成本](REBUTTAL_SELECTED_EVIDENCE.md#model-cost)。[原Hybrid留出](runs/004/REPORT.md)、[三模型E2E预测](responses/reviewer3/modeling_night151/README.md)、[增量与准备成本](responses/reviewer3/new_point_overheads_night161/snapshots/20261027_184651_final/README.md)。

措辞要点：实际硬件画像与Hybrid的未知shape泛化是两个层面；若篇幅紧，保留两组误差及这一区别，删除成本细节时仍保留“完整标签成本另计”。

## R4 / QboE：简单剪枝、搜索规模与Figure2

> We added Wanda and magnitude pruning under the same 2:4 structure and sparse backend on Llama-3.1-8B. SparseGPT achieves 42.92% MMLU and 21.44 WikiText word perplexity, versus 37.02%/38.27 for Wanda and 28.56%/516.33 for magnitude pruning. Their similar backend speeds help separate pruning quality from kernel acceleration and support SparseGPT as a useful sparse candidate in the cross-method policy space.
>
> We measured policy-search cost explicitly. After calibration, the two-stage exact surrogate solve takes 0.221–1.025 seconds on Qwen3, 0.192–1.986 seconds on Llama-3.1, and 0.196–0.494 seconds on Llama-2. This is solving time; profiling, label acquisition, export, and validation are reported separately.
>
> For kernel-level scalability, we additionally executed 112 GPU forwards at Qwen3-32B projection dimensions. These synthetic-weight measurements establish backend executability at larger shapes, rather than full-32B model quality or E2E performance. We also provide a font-independent vector replacement for Figure 2 to remove font-substitution dependencies.

对应[剪枝比较](REBUTTAL_SELECTED_EVIDENCE.md#pruning)与[求解成本](REBUTTAL_SELECTED_EVIDENCE.md#model-cost)、[112个大shape forward](runs/004/REPORT.md)、[Figure2替换PDF](runs/008/results/figure2_outlined.pdf)。

措辞要点：完整30/32B模型实验尚未完成，不能把上述112个kernel forward描述成满足了完整大模型评测要求。这里的搜索耗时为新增精确代理求解，不能改写成所有准备只需数秒。

## R5 / gkP9：NAS定位、接口、相同输入与反约束

> We agree that hardware-aware prediction and resource-constrained search have substantial precedents in NAS and mixed-precision optimization. Our contribution is their deployment-level realization for joint cross-backend, module-wise, and phase-wise allocation in a fixed LLM, including executable quantized/sparse representations and their runtime lifecycle. The expanded action space adds INT8, SmoothQuant INT8, and FP8 to the original five candidates.
>
> Internal weight formats are distinguished from activation interfaces: all 768 recorded Linear boundaries have BF16 inputs and outputs. Internal quantization/scaling is included in backend timing, while phase transfer and restoration are reported separately. For local perturbations, every candidate receives the same BF16-reference module-input tensor, and candidate outputs do not replace activations propagated through the reference backbone. Continuation features are a teacher-forced batching approximation; complete policies are evaluated separately through the actual KV runtime.
>
> The inverse formulation minimizes predicted quality loss under a latency cap and preserves the additive problem's optimal substructure; all eight forward/inverse consistency checks passed. A separate preregistered split analysis uses round 0 for guarded latency selection and rounds 1/2 for checking, with E2E and full-cycle caps handled separately. All 36 labels were feasible with no observed later-round overrun. These labels share 17 selected maps; this is finite-pool empirical evidence, not a production SLA guarantee.

对应[反约束与模块接口](REBUTTAL_SELECTED_EVIDENCE.md#interfaces-inverse)。[NAS定位](runs/008/NAS_CONTEXT.md)、[identical inputs实现](responses/reviewer5/overnight_run008/IDENTICAL_INPUTS.md)、[接口](responses/reviewer5/interfaces_run025/README.md)、[反约束](responses/reviewer5/solver_grid_run023/README.md)、[预注册实测校验](responses/reviewer5/latency_validation_night161/README.md)。

措辞要点：分割分析在三轮数据收集结束后执行，但规则与源码在首个输出前冻结，selection在读取后两轮前保存；不描述成第二轮采集前发生的在线干预。原对应点在这些cap下也没有超时，不宣称校验降低本批违约率；新cap与041原绝对cap不同。

## 精简建议

- R1保留预算满足率、同map速度与GPU/host成本。
- R2优先保留强基线、六负载与真实mixed路由；同候选代理例按剩余篇幅加入。
- R3保留两类预测误差、非Linear处理与校准成本口径。
- R4保留剪枝质量与真实求解耗时，简短说明规模范围和Figure2修复。
- R5保留NAS定位、BF16接口和反约束；identical inputs可配一条实现说明。

[完整精选表格](REBUTTAL_SELECTED_EVIDENCE.md) · [关联分析与来源审计](response174/analysis_audit.json)
