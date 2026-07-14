# Intermediate policy real-task quality plan

目标：对 stall-screened 的 prefill-decode 中间策略 point 34、36、37、38 测量真实生成任务分数，验证 WikiText NLL Pareto 趋势能否迁移到下游任务。

1. 复用 035 的连续 phase-hetero PMPD runner，适配 036 checkpoint/policy 路径；验证：每个点可被 runner 发现且每张 GPU 只执行一个 vLLM worker。
2. 仅用 GPU 0–4 测试 CNN/DM-1000、DialogSum-1500、IWSLT-333；验证：每个点/数据集样本覆盖完整、无重复或空输出。
3. 以同一指标器汇总 ROUGE-L/SacreBLEU，并与 035 的实测任务曲线并列；验证：输出明确标示中间点速度采用排除 >10 秒 stall 的 screened median。
