# Qwen3：考虑验证保护值的条件增量提案

030复用029条件加和模型及训练64窗口，不新增模型校准GPU作业。029开发validation显示仅优化均值的点可能被最终SE/余量筛掉，因此本轮在提案生成中预测128窗口保护值。使用配对anchor/组增量完整协方差，另加8个校准policy holdout的最大正残差；不是形式化置信保证。

固定残差余量=0.008279，提案guard目标为.005/.01/.02/.04，枚举65536组mask。最终池10个map，保留029全部条件均值提案和共同anchor；累计池不会人为丢掉mean-only的可行incumbent。模型/边际/协方差/余量未使用029 confirmation或030最终窗口重新拟合。

本轮128validation+128confirmation+16速度输入均为unused内部WikiText train窗口，排除原训练128、028的208和029的272窗口。不是全新官方语料。选择在confirmation前冻结，同mean+2SE+.005规则，未放宽ε。

## 同预算、同数据、同运行时比较

| Method | ε | Policy | Val guard | Confirm ΔNLL | Confirm guard | Mean pass | Guard pass | E2E × | Cycle × | Relative E2E /INT8 |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| mean_only | 0.01 | uniform_int8 | -0.00406 | -0.00846 | 0.00725 | True | True | 1.629 | 1.629 | 1.000 |
| mean_only | 0.04 | group_7fff | 0.01263 | 未确认 | 未确认 | None | None | 1.809 | 1.625 | 1.110 |
| guard_aware | 0.01 | group_3ee5 | 0.00180 | -0.00824 | 0.00773 | True | True | 1.712 | 1.581 | 1.050 |
| guard_aware | 0.04 | group_3ee5 | 0.00180 | -0.00824 | 0.00773 | True | True | 1.712 | 1.581 | 1.050 |

只对主版本和固定anchor开独立确认；mean-only若未与这些map重合，确认列留空，不冒充新测。两种方法的选择均在相同新validation上按真实guard和calibration E2E完成。mean-only选择也进入正式测速控制。GPU0/NUMA0、B8/S2048/O64、KV6GiB、2120/16384/[8]、3轮首尾BF16锚点，每轮3warmup+5timed，权重切换逐tensor相等检查通过。cycle含恢复，原始三轮保留，不把小差值称为显著。

## 完整生成

| Policy | CNN/DM | DialogSum | IWSLT | Avg score | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| group_3ee5 | 22.621 | 27.162 | 39.554 | 29.779 | 030 new Graph |
| phase_sq_int8_w4a16_ours | 22.714 | 27.062 | 39.988 | 29.921 | 027 same-map Graph |
| uniform_int8 | 22.747 | 26.822 | 39.586 | 29.718 | 028 same-map Graph |
| uniform_sq_int8 | 22.696 | 27.174 | 39.322 | 29.731 | 027 same-map Graph |

完整1000/1500/333是已有benchmark上的冻结策略评测，不反向选择配置；同map/代码/数据协议的027/028 Graph分数明确继承，不混入旧eager。Avg是ROUGE-L%/BLEU算术摘要，不是NLL约束。本轮未重跑四项MC。

## 边界

- 经验残差只有8个policy holdout，协方差只有64个训练窗口；保护值预测仍可能失准，必须看新confirmation的均值与保护值，不宣称任意分布保证。
- 仅固定SQ-prefill的16组子空间，不是全144模块/全部8候选全局最优。旧结果、失败和均值提案均保留。
- 不读取confirmation来改模型、窗口数、预算或重选；不将相同map的多预算标签计为独立策略。

[预测guard与真实validation](results/forecast_vs_validation.csv) · [同预算比较](results/same_budget_comparison.csv) · [三轮速度](results/formal_speed.csv) · [完整生成](results/full_generation.csv) · [残差来源](results/guard_forecast.json)
