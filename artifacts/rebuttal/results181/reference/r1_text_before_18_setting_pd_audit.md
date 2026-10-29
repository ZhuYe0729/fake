# R1质量预算：P+D文字回复草稿

统计单位为“给定预算下选出的完整策略”，每点用64个held-out文本窗口的平均ΔNLL判断；不以逐文本超限比例替代策略均值超限率。以下只报告既有P+D审计，不新增实验。

当前原DP四个预算的均值超限为0/4（满足率4/4）；另一MILP求解器版本的均值超限为1/4（满足率3/4），只发生在这四点的最紧预算0.005，实测0.00574726、预测0.00470348、超限0.00074726。两个版本分别报告，不合并频率。

原DP四个选点的MAE=0.00764107；论文Llama-3.1留出策略MAE=0.04125365、Spearman=0.96284830。不同模型/集合只作背景，不能证明新测试更优、最优策略选择不受影响或紧预算硬保证。

可以解释：当ε很小时，固定绝对预测残差和有限样本不确定性相对于ε更突出。单个最紧预算点的超限不足以证明违约频率随预算收紧而上升；其配对SE约0.00713，也不能单凭这一数值认定超限主要由抽样造成。NLL差异小与下游任务等效是不同结论，后者需要对应策略的任务证据。

## 适用于原DP的英文正文

> We distinguish feasibility under the predicted quality model from satisfaction of the measured budget. In an additional Qwen3-8B prefill–decode audit (B=8, input/output lengths 2048/64), we evaluated policies returned by the original DP solver with the seven-candidate extension for ε=0.005, 0.01, 0.02, and 0.04 on 64 held-out WikiText windows per policy. All four measured mean ΔNLL values were within budget (0/4 observed mean-budget violations); the mean absolute prediction error across these four selected policies was 0.00764 nats/token. These results are specific to this audit and do not establish a zero violation probability.
>
> The paper reports a held-out ΔNLL MAE of 0.0413 and Spearman correlation of 0.963 on Llama-3.1, supporting useful overall prediction and ranking rather than exact budget certification. Under very tight budgets, absolute surrogate residuals and finite-sample uncertainty can be comparable to ε. We therefore regard the quality constraint as an empirical predictive constraint; the additional measured-validation procedure provides a further check when strict satisfaction is required.

## 仅在同时说明附加MILP审计时可追加

> In a separate audit using a MILP solver for the same frozen surrogate, one of four P+D policies exceeded its budget. This occurred at the tightest evaluated budget (ε=0.005): predicted ΔNLL was 0.004703 and measured mean ΔNLL was 0.005747, an excess of 0.000747 nats/token. This is a solver-variant result, not the violation rate of the original DP.

[预算点及逐点来源](../pd_quality_budget/README.md) · [CSV](../pd_quality_budget/points.csv) · [质量模型参数身份复核](../../../results181/reference/pd_quality_model_identity.json) · [旧三模型validation增强版](../../../response174/budget_counts.csv)
