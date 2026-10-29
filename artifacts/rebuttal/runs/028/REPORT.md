# Qwen3：基线保留与部分 W4A16 decode 选点修正

本轮沿用原量化/runtime、冻结训练代理、ε及 mean+2SE+.005 规则，新增24个固定候选和更多内部留出验证窗口。128窗口决定选择；先冻结选择，再打开64窗口确认。新窗口排除了原128个训练/激活校准样本的任何token区间，与所有新split也不重叠；它们来自原WikiText train语料，不称为全新官方测试集。

## 选择与独立确认

| ε | 选点目标 | 选中配置 | Val guard | Confirm ΔNLL ±2SE | Mean pass | Guard pass | E2E × | Cycle × |
| ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 0.01 | e2e_ms | uniform_sq_int8 | -0.00935 | -0.02153 ± 0.01230 | True | True | 1.632 | 1.632 |
| 0.01 | cycle_ms | uniform_sq_int8 | -0.00935 | -0.02153 ± 0.01230 | True | True | 1.632 | 1.632 |
| 0.04 | e2e_ms | phase_sq_int8_w4a16_ours | 0.02941 | 0.03018 ± 0.01423 | True | False | 1.864 | 1.651 |
| 0.04 | cycle_ms | phase_sq_int8_w4a16_ours | 0.02941 | 0.03018 ± 0.01423 | True | False | 1.864 | 1.651 |

Mean pass只表示64窗口平均值未超预算；Guard pass使用相同mean+2SE+.005，二者分别列出。若平均通过但guard未通过，不能声称确认了高置信度可行性；确认结果不反向重选。E2E为原主目标，cycle是实验前另声明的含恢复目标；本轮若两者选同一map，重复记录不是不同配置。正式速度为GPU0/NUMA0、B8/S2048/O64、6GiB KV、native decode Graph，max_model_len=2120、max_num_batched_tokens=16384、capture_sizes=[8]，3轮，每轮3 warmup+5 timed，首尾BF16锚点平均，最终取三轮均值。沿用016 benchmark协议；019为支持六负载使用4168/32768/[1,8]，不把两套campaign混成同轮速度。真实权重切换逐tensor精确相等检查通过。

## 同轮速度对照

| 配置 | E2E × | Cycle × | TTFT ms | TPOT ms | Restore ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform_dense_bf16 | 1.000 | 1.000 | 1136.53 | 13.719 | 0.15 |
| uniform_int8 | 1.633 | 1.633 | 604.80 | 9.867 | 0.17 |
| phase_sq_int8_w4a16_ours | 1.864 | 1.651 | 606.72 | 7.421 | 138.24 |
| uniform_sq_int8 | 1.632 | 1.632 | 606.53 | 9.855 | 0.15 |
| phase_dense_nvfp4_sq_int8 | 1.554 | 1.455 | 670.54 | 9.802 | 88.13 |

## 完整生成任务

| 配置 | 来源 | CNN/DM | DialogSum | IWSLT | Avg score |
| --- | --- | ---: | ---: | ---: | ---: |
| phase_sq_int8_w4a16_ours | 027 same-map Graph | 22.714 | 27.062 | 39.988 | 29.921 |
| uniform_int8 | 028 new Graph | 22.747 | 26.822 | 39.586 | 29.718 |
| uniform_sq_int8 | 027 same-map Graph | 22.696 | 27.174 | 39.322 | 29.731 |

完整1000/1500/333输入为已有benchmark；分数只描述冻结选点，不参与选择。Avg是ROUGE-L%和BLEU的算术摘要，不替代NLL约束。相同prefill的选择题质量可从原均匀配置单列引用，本轮未重跑四项MC。

## 固定窗口顺序的保护值敏感性

| 配置 | 前32窗口 guard | 前64窗口 guard | 全128窗口 guard（用于选择） |
| --- | ---: | ---: | ---: |
| uniform_int8 | 0.01583 | 0.00266 | -0.00173 |
| uniform_sq_int8 | 0.00394 | -0.00654 | -0.00935 |
| repair_sq_int8_w4_008 | 0.01727 | 0.01135 | 0.01249 |
| phase_sq_int8_w4a16_ours | 0.02356 | 0.02273 | 0.02941 |

前缀表只分析同一新数据集的样本量敏感性；旧验证语料窗口与本轮内部train留出不同，不能把旧/新差异全部归因于窗口数量。


## 本次修正解决的内容

- 显式保留普通INT8、SQ-INT8及相关阶段组合，加入从基线出发的嵌套decode局部候选。训练代理仅决定替换顺序，不提前排除质量可行的基线。
- 只按固定128窗口规则筛选，32/64窗口保护值另列用于样本量敏感性；不挑最有利样本量。
- 同一可行池内，用实测calibration E2E或cycle选最小者，断言不慢于可行均匀INT8 incumbent。该保证属于calibration候选池，不是所有工作负载/任务/新确认数据上的必胜保证。
- 旧精确解已经选中过全W4A16但真实NLL失败；本轮不将该失败改写为ε=.01成功，不改旧主表。

原代理自身holdout RMSE为0.02740，高于.01预算；SQ-INT8/INT8/W4A16的decode o_proj代理贡献各有36/36/36个为0。这是模型近似，不表示真实量化误差为0。本轮没有重训或修改该代理，也未证明局部候选精化能改善其预测。

这是一轮围绕两种INT8的有界局部精化，不是原全部8候选/所有历史MILP点的全局重新求解。不得将本轮ε=.04点宣称为优于原016所有ε=.04解；旧点、旧失败和本轮未入选候选均保留。

[全部24个候选](results/all_validation_candidates.csv) · [选择结果](results/selected_results.csv) · [候选池消融](results/candidate_pool_ablation.csv) · [三轮速度](results/formal_campaigns.csv) · [完整生成](results/full_generation.csv) · [旧选点原因](reference/old_selection_diagnosis.json)
