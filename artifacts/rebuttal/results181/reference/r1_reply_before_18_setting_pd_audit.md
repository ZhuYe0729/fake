## R1 / AK6a：预算满足率与双表示开销

> We distinguish feasibility under the predicted quality model from satisfaction of the measured budget. In an additional Qwen3-8B prefill–decode audit (B=8, input/output lengths 2048/64), we evaluated policies returned by the original DP solver with the seven-candidate extension for ε=0.005, 0.01, 0.02, and 0.04 on 64 held-out WikiText windows per policy. All four measured mean ΔNLL values were within budget (0/4 observed mean-budget violations); the mean absolute prediction error across these four selected policies was 0.00764 nats/token. These results are specific to this audit and do not establish a zero violation probability.
>
> The paper reports a held-out ΔNLL MAE of 0.0413 and Spearman correlation of 0.963 on Llama-3.1, supporting useful overall prediction and ranking rather than exact budget certification. Under very tight budgets, absolute surrogate residuals and finite-sample uncertainty can be comparable to ε. We therefore regard the quality constraint as an empirical predictive constraint; the additional measured-validation procedure provides a further check when strict satisfaction is required.
>
> We also quantify representation storage. In the dual-resident mixed-runtime experiment, a representative Qwen policy has 17.104 GiB peak Torch-allocated GPU memory versus 21.384 GiB for BF16, together with an additional 8.236 GiB host adapter-weight cache. Thus, we report both compressed GPU residency and the host-side cost instead of treating phase switching as free.

对应[P+D文字回复与口径](responses/reviewer1/quality_budget_text_reply/README.md)、[历史直接求解审计](responses/reviewer1/pd_quality_budget/README.md)与[运行开销](REBUTTAL_SELECTED_EVIDENCE.md#runtime-cost)。

措辞要点：0/4仅指这批Qwen P+D原DP策略的测试均值，不能泛化到Prefill、其他模型或真实违约概率；0.00764是四个选点的绝对预测MAE，0.0413/0.963来自论文Llama-3.1留出策略，集合不同。没有证据将全部偏差归因于样本，或证明违约仅在小预算出现、对策略选择无影响、下游任务等效。额外实测validation属于补充流程，旧24标签结果仍完整保留在精选证据折叠区。

