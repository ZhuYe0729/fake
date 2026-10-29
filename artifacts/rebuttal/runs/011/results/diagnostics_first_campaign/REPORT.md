# Run011：验证后策略与强化HAWQ的同卡标准负载复核

005验证后策略与009强化基线此前分别在GPU2/GPU0计时。本轮把009全部35比较点与005全部8个质量预算验证选择合并，同一GPU0计时；19份唯一map、23个唯一policy/scenario，所有43个标签保留。没有新策略、质量拟合或规则调整。

质量直接继承005/009同64个test窗口、B8 prompt-logprobs/真实KV teacher forcing的结果，逐SHA核验，重复map的均值也核对一致。所有canonical/scales/kernel相同。这里新增的是确认速度，不把再次计时称为新的独立质量测试。

速度为B8/S2048/O1或64、eager、无chunked/prefix cache、6GiB KV、2 warmups+5 timed，统一新seed144983；方法顺序用固定seed144985打乱，BF16开头/结尾各一次，合并10个timed burst作锚。

| 场景 | BF16起点ms | 终点ms | 终点/起点 | 合并10轮ms |
| --- | ---: | ---: | ---: | ---: |
| prefill_only | 1126.9 | 1130.7 | 1.003 | 1130.6 |
| prefill_decode | 2016.9 | 2013.5 | 0.998 | 2015.0 |

| 场景 | 比较点 | 实测ΔNLL | 描述性95%窗口区间 | E2E ms | 相对BF16 |
| --- | --- | ---: | --- | ---: | ---: |
| prefill_only | uniform_dense_bf16 | 0.00000 | [0.00000, 0.00000] | 1130.6 | 1.000× |
| prefill_only | uniform_int8 | -0.00699 | [-0.01044, -0.00356] | 597.3 | 1.893× |
| prefill_only | uniform_sq_int8 | -0.01252 | [-0.01539, -0.00964] | 600.0 | 1.884× |
| prefill_only | uniform_fp8 | 0.00633 | [0.00513, 0.00754] | 670.0 | 1.687× |
| prefill_only | transfer_daso8_prefill_only_02 | 0.03606 | [0.03090, 0.04160] | 572.5 | 1.975× |
| prefill_only | hawq_marlin_int4_signed_prefill_only_q0.01 | 0.01011 | [0.00738, 0.01273] | 657.6 | 1.719× |
| prefill_only | hawq_marlin_int4_signed_prefill_only_q0.07 | -0.00699 | [-0.01044, -0.00356] | 597.3 | 1.893× |
| prefill_only | hawq_fp4_phase_pair_signed_prefill_only_q0.01 | 0.01482 | [0.01156, 0.01791] | 654.6 | 1.727× |
| prefill_only | hawq_fp4_phase_pair_signed_prefill_only_q0.07 | 0.00900 | [0.00417, 0.01371] | 516.6 | 2.189× |
| prefill_only | hawq_marlin_int4_clip_zero_prefill_only_q0.01 | 0.01474 | [0.01194, 0.01765] | 657.1 | 1.721× |
| prefill_only | hawq_marlin_int4_clip_zero_prefill_only_q0.07 | -0.00699 | [-0.01044, -0.00356] | 597.3 | 1.893× |
| prefill_only | hawq_fp4_phase_pair_clip_zero_prefill_only_q0.01 | 0.01302 | [0.01043, 0.01564] | 656.5 | 1.722× |
| prefill_only | hawq_fp4_phase_pair_clip_zero_prefill_only_q0.07 | 0.00900 | [0.00417, 0.01371] | 516.6 | 2.189× |
| prefill_only | hawq_marlin_int4_absolute_prefill_only_q0.01 | 0.01356 | [0.01042, 0.01682] | 656.0 | 1.724× |
| prefill_only | hawq_marlin_int4_absolute_prefill_only_q0.07 | -0.00699 | [-0.01044, -0.00356] | 597.3 | 1.893× |
| prefill_only | hawq_fp4_phase_pair_absolute_prefill_only_q0.01 | 0.01356 | [0.01042, 0.01682] | 656.0 | 1.724× |
| prefill_only | hawq_fp4_phase_pair_absolute_prefill_only_q0.07 | 0.00900 | [0.00417, 0.01371] | 516.6 | 2.189× |
| prefill_only | verified_prefill_only_q0.005 | -0.01252 | [-0.01539, -0.00964] | 600.0 | 1.884× |
| prefill_only | verified_prefill_only_q0.01 | -0.01252 | [-0.01539, -0.00964] | 600.0 | 1.884× |
| prefill_only | verified_prefill_only_q0.02 | -0.01252 | [-0.01539, -0.00964] | 600.0 | 1.884× |
| prefill_only | verified_prefill_only_q0.04 | 0.01936 | [0.01371, 0.02531] | 514.0 | 2.200× |
| prefill_decode | uniform_dense_bf16 | 0.00000 | [0.00000, 0.00000] | 2015.0 | 1.000× |
| prefill_decode | uniform_int8 | -0.00819 | [-0.02016, 0.00378] | 2187.2 | 0.921× |
| prefill_decode | uniform_sq_int8 | -0.00577 | [-0.01675, 0.00589] | 2734.8 | 0.737× |
| prefill_decode | uniform_fp8 | 0.01208 | [0.00677, 0.01723] | 2280.2 | 0.884× |
| prefill_decode | transfer_daso5_measured_prefill_decode_04 | -0.00356 | [-0.02014, 0.01171] | 1615.5 | 1.247× |
| prefill_decode | transfer_daso8_prefill_decode_01 | -0.00343 | [-0.01857, 0.01140] | 1679.7 | 1.200× |
| prefill_decode | hawq_marlin_int4_signed_prefill_decode_q0.01 | -0.00302 | [-0.01514, 0.00990] | 2116.4 | 0.952× |
| prefill_decode | hawq_marlin_int4_signed_prefill_decode_q0.07 | -0.00302 | [-0.01514, 0.00990] | 2116.4 | 0.952× |
| prefill_decode | hawq_fp4_phase_pair_signed_prefill_decode_q0.01 | -0.01591 | [-0.03063, -0.00145] | 1741.6 | 1.157× |
| prefill_decode | hawq_fp4_phase_pair_signed_prefill_decode_q0.07 | 0.00450 | [-0.00964, 0.01849] | 1443.1 | 1.396× |
| prefill_decode | hawq_marlin_int4_clip_zero_prefill_decode_q0.01 | -0.00302 | [-0.01514, 0.00990] | 2116.4 | 0.952× |
| prefill_decode | hawq_marlin_int4_clip_zero_prefill_decode_q0.07 | -0.00302 | [-0.01514, 0.00990] | 2116.4 | 0.952× |
| prefill_decode | hawq_fp4_phase_pair_clip_zero_prefill_decode_q0.01 | -0.01591 | [-0.03063, -0.00145] | 1741.6 | 1.157× |
| prefill_decode | hawq_fp4_phase_pair_clip_zero_prefill_decode_q0.07 | 0.00450 | [-0.00964, 0.01849] | 1443.1 | 1.396× |
| prefill_decode | hawq_marlin_int4_absolute_prefill_decode_q0.01 | -0.00302 | [-0.01514, 0.00990] | 2116.4 | 0.952× |
| prefill_decode | hawq_marlin_int4_absolute_prefill_decode_q0.07 | -0.00302 | [-0.01514, 0.00990] | 2116.4 | 0.952× |
| prefill_decode | hawq_fp4_phase_pair_absolute_prefill_decode_q0.01 | -0.01591 | [-0.03063, -0.00145] | 1741.6 | 1.157× |
| prefill_decode | hawq_fp4_phase_pair_absolute_prefill_decode_q0.07 | 0.00450 | [-0.00964, 0.01849] | 1443.1 | 1.396× |
| prefill_decode | verified_prefill_decode_q0.005 | 0.00000 | [0.00000, 0.00000] | 2015.0 | 1.000× |
| prefill_decode | verified_prefill_decode_q0.01 | 0.00000 | [0.00000, 0.00000] | 2015.0 | 1.000× |
| prefill_decode | verified_prefill_decode_q0.02 | 0.00633 | [-0.00009, 0.01281] | 1742.5 | 1.156× |
| prefill_decode | verified_prefill_decode_q0.04 | -0.00593 | [-0.02175, 0.00939] | 1692.7 | 1.190× |

## 适用边界

- DASO+validation是原算法之外新增的候选实测验证阶段，其额外离线成本约38.7分钟wall（该次三GPU并行选择，含I/O/初始化）见005；不能将效果归于原文已有质量保证。
- .005/.01/.02/.04的8个验证预算点包含重复选择。很紧的P+D预算回退BF16；另有预算.04点在test均值低于.01，也不能将它称为部署前epsilon=.01验证规则选出的策略。
- 质量上限切片是已测点的事后Pareto比较。相同实际ΔNLL上限不等于同一预测约束、同一准备成本或全局最优；全部Hessian sign/clip/abs变体与min-cost clamp明确保留。
- BF16起终点、所有计时min/max以及使用各端点归一化的速度均归档；小差距和时间漂移不能解释成统计显著优势。
- HAWQ为PTQ精度分配移植，不是原CNN QAT/TVM全栈复现。真实mixed调度另见006，完整下游质量另见010；本轮没有30B整模型实验。

![统一标准负载](results/figures/matched_standard.png)

[全部43点CSV](results/summary_final.csv) · [全部质量上限切片](results/quality_matched_frontiers.json) · [冻结比较池](configs/candidate_pool.json) · [质量来源](reference/inherited_quality.json)。
