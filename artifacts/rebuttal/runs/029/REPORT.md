# Qwen3：固定prefill的条件增量质量模型

本轮为DASO可选质量代理分支：固定SQ-INT8 prefill，16个深度/类型组，每组9个Linear，decode组选择SQ-INT8或W4A16。相同真实Graph/KV路径采集配对整模型NLL，未改量化/kernel、预算或mean+2SE+.005门控。原质量模型与封存主表保留。

校准采用原训练数据前64窗口；26个拟合policy与8个未拟合组合预先固定。校准policy holdout用于模型选择，不是新语料测试。128 validation/128 confirmation来自token-disjoint内部WikiText train窗口，排除原128训练与028全部208窗口；不是新官方benchmark语料。

## 代理预测诊断

| Model | Policy-holdout MAE | Policy-holdout RMSE | Fresh-validation MAE | Fresh-validation RMSE |
| --- | ---: | ---: | ---: | ---: |
| old_anchored | 0.01497 | 0.01975 | 0.02787 | 0.03066 |
| refit_original | 0.01349 | 0.01710 | 0.02514 | 0.02926 |
| conditional_additive | 0.00220 | 0.00330 | 0.01030 | 0.01054 |
| conditional_pair | 0.00336 | 0.00484 | 0.01688 | 0.01721 |

主版本 **conditional_additive** 仅由8个预定policy holdout的MAE选定，早于最终validation；不根据后者或confirmation反选模型。refit_original保留原架构/超参，使用相同26个新标签；old_anchored只纠正基线偏置。其余两模型直接校准signed分组增量，pair版额外包含8个预定交互。

## 冻结主版本与独立确认

| ε | Policy | W4 decode /144 | Val guard | Confirm ΔNLL ±2SE | Confirm guard | Mean pass | Guard pass | E2E × | Cycle × | Relative E2E /INT8 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 0.01 | uniform_sq_int8 | 0 | 0.00262 | -0.01036 ± 0.00820 | 0.00284 | True | True | 1.630 | 1.630 | 1.001 |
| 0.04 | group_7fff | 135 | 0.02326 | 0.00084 ± 0.01377 | 0.01961 | True | True | 1.796 | 1.607 | 1.103 |

确认均值与保护值分列；任何失败均不反向重选。正式速度GPU0/NUMA0、B8/S2048/O64、KV6GiB、2120/16384/[8]、三轮BF16首尾锚点、每轮3warmup+5timed，权重切换逐tensor相等检查通过。三轮均值/原始分项全部保留，未将微小数值差称作统计显著。cycle包含恢复，E2E延续原边界。

## 完整生成

| Policy | CNN/DM | DialogSum | IWSLT | Avg score | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| group_7fff | 22.776 | 27.166 | 39.698 | 29.880 | 029 new Graph |
| phase_sq_int8_w4a16_ours | 22.714 | 27.062 | 39.988 | 29.921 | 027 same-map Graph |
| uniform_int8 | 22.747 | 26.822 | 39.586 | 29.718 | 028 same-map Graph |
| uniform_sq_int8 | 22.696 | 27.174 | 39.322 | 29.731 | 027 same-map Graph |

1000/1500/333完整benchmark只描述冻结选择，不参与训练/模型选择。相同map、代码和数据协议的027/028 Graph分数带SHA复用；普通INT8使用028当前Graph分数，不混入002旧eager。Avg为ROUGE-L%/BLEU混合算术摘要，不是质量约束；本轮未重跑四项MC。

## 适用范围与成本

- 36个校准map×64窗口，其中26个用于拟合、8个用于模型选择，另外固定控制单列；新增标签成本不能省略。最终validation池为7个唯一map。
- 各模型仅在共同16组子空间枚举65536个mask；目标为0/.005/.01/.04，不声称144模块全部8候选的全局最优。各模型的部署选点只使用本模型提案+共同anchor。
- 配对交互是有限近似，不表达任意跨层传播；更低代理误差不自动等于更快部署或所有生成分数更高。可行INT8 anchor始终保留。
- 新鲜validation数据只评估/选配置，confirmation只确认；旧结果不覆盖。所有参数、失败和不利结果保留。

[逐组真实增量](results/group_increment_diagnostic.csv) · [policy holdout](results/policy_holdout_predictions.csv) · [新validation误差](results/fresh_validation_prediction_metrics.csv) · [选点](results/selected_results.csv) · [速度](results/formal_speed.csv) · [完整生成](results/full_generation.csv)
