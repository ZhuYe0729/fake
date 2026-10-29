# SQ-INT8 prefill / W4A16 decode：固定对照与完整生成结果

每任务128个固定seed抽样输入，四配置全部新跑相同greedy/256上限/6GiB KV/native decode Graph。前表为已评估benchmark的探索性子集，不作为完整任务结果、独立确认集或统计显著性结论；后文另列新组合与SQ对照的完整生成。新组合与Uniform SQ-INT8固定相同prefill；与旧DASO点比较时两个阶段均改变。

| Method | CNN/DM ROUGE-L | DialogSum ROUGE-L | IWSLT BLEU | Generation Avg score |
| --- | ---: | ---: | ---: | ---: |
| SQ-INT8 prefill / W4A16 decode | 22.611 | 24.350 | 41.277 | 29.412 |
| Uniform SQ-INT8 | 22.240 | 24.344 | 40.924 | 29.169 |
| Uniform INT8 | 22.249 | 24.252 | 40.627 | 29.042 |
| Old DASO-7 epsilon=.01 (NVFP4 / SQ-INT8) | 21.812 | 24.175 | 40.433 | 28.807 |

平均分为ROUGE-L%与SacreBLEU的混合算术摘要；原始分项全部保留，不作为统一准确率。

## 中心负载速度（复用019同一map）

B8/S2048/O64，GPU0/NUMA0。各轮自身BF16首尾锚点，3warmup+5timed；表中注明实际可用campaign数，cycle包含恢复。生成耗时不当作基准速度。

| Policy | Campaigns | E2E × | Cycle × | TTFT ms | TPOT ms | Restore ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sq_int8_w4a16 | 3 | 1.741 | 1.569 | 611.145 | 8.359 | 124.319 |
| uniform_sq_int8 | 3 | 1.636 | 1.636 | 606.589 | 9.597 | 0.133 |
| old_daso7_e001 | 3 | 1.551 | 1.460 | 671.558 | 9.793 | 71.444 |

## 六负载固定配置迁移：相对全SQ-INT8

全部六负载均来自019三轮。新组合与全SQ-INT8固定map比较，未逐负载重新搜索；数值大于1表示新组合更快。

| B | S | O | Relative E2E speedup | Relative cycle speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2048 | 64 | 1.064 | 0.959 |
| 1 | 2048 | 128 | 1.099 | 0.975 |
| 8 | 512 | 128 | 1.320 | 1.151 |
| 8 | 4096 | 64 | 1.077 | 1.010 |
| 8 | 512 | 256 | 1.336 | 1.241 |
| 1 | 512 | 256 | 1.388 | 1.278 |

## 独立负载NLL记录（019同64窗口）

| Policy | Measured ΔNLL | Paired ±2SE | Mean exceeds .01 |
| --- | ---: | ---: | --- |
| sq_int8_w4a16 | 0.02009 | 0.01376 | True |
| uniform_sq_int8 | -0.03040 | 0.01334 | False |
| old_daso7_e001 | -0.04388 | 0.01406 | False |

新组合是人工固定phase对照，不因生成子集改善而自动成为满足ε=.01的DASO选点。NLL约束与生成分数分别报告。

## 历史完整生成参考（不同运行时，单列）

002为旧eager，015为旧DASO的Graph；以下完整生成只作历史参考，与上述128条新pilot分开。

| Policy | CNN/DM 1000 | DialogSum 1500 | IWSLT 333 |
| --- | ---: | ---: | ---: |
| sq_int8_w4a16 | 22.733 | 27.063 | 39.988 |
| uniform_sq_int8 | 22.697 | 27.179 | 39.322 |
| uniform_int8 | 22.787 | 27.275 | 39.795 |
| old_daso7_e001 | 22.451 | 26.871 | 39.579 |

## 新组合完整三任务与归档完整对照

新组合与全SQ-INT8对照的1000/1500/333均在027 Graph新测。其余两行完整分数来自明确标注的历史归档；四配置另有上面的同128条新Graph pilot及逐token历史匹配计数，不把历史完整对照称为本轮全量重测。

| Policy | Runtime source | CNN/DM1000 | DialogSum1500 | IWSLT333 | Generation Avg score |
| --- | --- | ---: | ---: | ---: | ---: |
| sq_int8_w4a16 | 027 new Graph | 22.714 | 27.062 | 39.988 | 29.921 |
| uniform_sq_int8 | 027 new Graph | 22.696 | 27.174 | 39.322 | 29.731 |
| uniform_int8 | 002 archived full generation | 22.787 | 27.275 | 39.795 | 29.952 |
| old_daso7_e001 | 015 archived full generation | 22.451 | 26.871 | 39.579 | 29.634 |

普通INT8的旧015完整主表E2E为1.645×，属另一campaign参考，未冒充上述同轮测量。本pilot的四项选择题不重跑：新配置prefill等价于Uniform SQ-INT8，原完整MC结果另有来源。

[生成CSV](results/generation_pilot.csv) · [相对各对照的差值](results/candidate_minus_controls.csv) · [冻结设计与抽样索引](configs/design.json) · [来源速度019](../019/README.md)
