# 预先固定的实测延迟校验（R5 gkP9 / R3 LrT2）

041三轮原18点完整保留，零新增GPU测量。首个速度输出前冻结规则/源码SHA，round0校验，round1/2共16个timed样本检查。分析在原始测量收集完后执行：selection先于confirmation读取保存，但不是第二轮采集前发生的在线干预。

按场景/阶段/分配器分六池，每池只有三个既有incumbent。物理cap=r×round0该场景BF16首尾16样本中位数，E2E/cycle分别定义；与原041绝对E2E cap不同，不能用新cap宣称旧上限通过。候选round0最大值×1.05须通过；再按各自原solver objective、校验中位数、policy id选择。无通过点明确不可行，不能用更慢BF16冒充满足速度预算。

36个标签中36个可行，0个不可行；可行标签涉及17个唯一map。重复选点和复用测试样本不视为独立成功次数。5%为固定经验余量，不提供覆盖定理或生产SLA保证；ΔNLL复用旧64个已查看窗口。

| 场景/阶段 | 分配器 | 指标 | r | cap ms | 选择 | 可行候选 | 原点超限/16 | 校验后超限/16 | 选择ΔNLL |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| prefill_decode/phase_pair | daso | e2e_ms | 0.95 | 1866.64 | daso_prefill_decode_phase_pair_r0.95 | 3 | 0 | 0 | 0.00090 |
| prefill_decode/phase_pair | daso | e2e_ms | 0.80 | 1571.91 | daso_prefill_decode_phase_pair_r0.8 | 2 | 0 | 0 | 0.00345 |
| prefill_decode/phase_pair | daso | e2e_ms | 0.65 | 1277.18 | daso_prefill_decode_phase_pair_r0.65 | 1 | 0 | 0 | 0.01499 |
| prefill_decode/phase_pair | daso | cycle_ms | 0.95 | 1866.77 | daso_prefill_decode_phase_pair_r0.95 | 3 | 0 | 0 | 0.00090 |
| prefill_decode/phase_pair | daso | cycle_ms | 0.80 | 1572.02 | daso_prefill_decode_phase_pair_r0.8 | 2 | 0 | 0 | 0.00345 |
| prefill_decode/phase_pair | daso | cycle_ms | 0.65 | 1277.27 | daso_prefill_decode_phase_pair_r0.65 | 1 | 0 | 0 | 0.01499 |
| prefill_decode/phase_pair | hawq_signed | e2e_ms | 0.95 | 1866.64 | hawq_signed_prefill_decode_phase_pair_r0.8 | 2 | 0 | 0 | 0.00794 |
| prefill_decode/phase_pair | hawq_signed | e2e_ms | 0.80 | 1571.91 | hawq_signed_prefill_decode_phase_pair_r0.8 | 2 | 0 | 0 | 0.00794 |
| prefill_decode/phase_pair | hawq_signed | e2e_ms | 0.65 | 1277.18 | hawq_signed_prefill_decode_phase_pair_r0.65 | 1 | 0 | 0 | 0.01418 |
| prefill_decode/phase_pair | hawq_signed | cycle_ms | 0.95 | 1866.77 | hawq_signed_prefill_decode_phase_pair_r0.8 | 2 | 0 | 0 | 0.00794 |
| prefill_decode/phase_pair | hawq_signed | cycle_ms | 0.80 | 1572.02 | hawq_signed_prefill_decode_phase_pair_r0.8 | 2 | 0 | 0 | 0.00794 |
| prefill_decode/phase_pair | hawq_signed | cycle_ms | 0.65 | 1277.27 | hawq_signed_prefill_decode_phase_pair_r0.65 | 1 | 0 | 0 | 0.01418 |
| prefill_decode/same_operator | daso | e2e_ms | 0.95 | 1866.64 | daso_prefill_decode_same_operator_r0.95 | 3 | 0 | 0 | 0.00002 |
| prefill_decode/same_operator | daso | e2e_ms | 0.80 | 1571.91 | daso_prefill_decode_same_operator_r0.8 | 2 | 0 | 0 | 0.00698 |
| prefill_decode/same_operator | daso | e2e_ms | 0.65 | 1277.18 | daso_prefill_decode_same_operator_r0.65 | 1 | 0 | 0 | 0.01319 |
| prefill_decode/same_operator | daso | cycle_ms | 0.95 | 1866.77 | daso_prefill_decode_same_operator_r0.95 | 3 | 0 | 0 | 0.00002 |
| prefill_decode/same_operator | daso | cycle_ms | 0.80 | 1572.02 | daso_prefill_decode_same_operator_r0.8 | 2 | 0 | 0 | 0.00698 |
| prefill_decode/same_operator | daso | cycle_ms | 0.65 | 1277.27 | daso_prefill_decode_same_operator_r0.65 | 1 | 0 | 0 | 0.01319 |
| prefill_decode/same_operator | hawq_signed | e2e_ms | 0.95 | 1866.64 | hawq_signed_prefill_decode_same_operator_r0.95 | 3 | 0 | 0 | 0.00742 |
| prefill_decode/same_operator | hawq_signed | e2e_ms | 0.80 | 1571.91 | hawq_signed_prefill_decode_same_operator_r0.8 | 2 | 0 | 0 | 0.00958 |
| prefill_decode/same_operator | hawq_signed | e2e_ms | 0.65 | 1277.18 | hawq_signed_prefill_decode_same_operator_r0.65 | 1 | 0 | 0 | 0.01812 |
| prefill_decode/same_operator | hawq_signed | cycle_ms | 0.95 | 1866.77 | hawq_signed_prefill_decode_same_operator_r0.95 | 3 | 0 | 0 | 0.00742 |
| prefill_decode/same_operator | hawq_signed | cycle_ms | 0.80 | 1572.02 | hawq_signed_prefill_decode_same_operator_r0.8 | 2 | 0 | 0 | 0.00958 |
| prefill_decode/same_operator | hawq_signed | cycle_ms | 0.65 | 1277.27 | hawq_signed_prefill_decode_same_operator_r0.65 | 1 | 0 | 0 | 0.01812 |
| prefill_only/same_operator | daso | e2e_ms | 0.95 | 1053.01 | daso_prefill_only_same_operator_r0.95 | 3 | 0 | 0 | 0.00930 |
| prefill_only/same_operator | daso | e2e_ms | 0.80 | 886.75 | daso_prefill_only_same_operator_r0.8 | 2 | 0 | 0 | 0.00908 |
| prefill_only/same_operator | daso | e2e_ms | 0.65 | 720.48 | daso_prefill_only_same_operator_r0.65 | 1 | 0 | 0 | 0.00860 |
| prefill_only/same_operator | daso | cycle_ms | 0.95 | 1053.16 | daso_prefill_only_same_operator_r0.95 | 3 | 0 | 0 | 0.00930 |
| prefill_only/same_operator | daso | cycle_ms | 0.80 | 886.87 | daso_prefill_only_same_operator_r0.8 | 2 | 0 | 0 | 0.00908 |
| prefill_only/same_operator | daso | cycle_ms | 0.65 | 720.59 | daso_prefill_only_same_operator_r0.65 | 1 | 0 | 0 | 0.00860 |
| prefill_only/same_operator | hawq_signed | e2e_ms | 0.95 | 1053.01 | hawq_signed_prefill_only_same_operator_r0.95 | 3 | 0 | 0 | 0.00670 |
| prefill_only/same_operator | hawq_signed | e2e_ms | 0.80 | 886.75 | hawq_signed_prefill_only_same_operator_r0.8 | 2 | 0 | 0 | 0.00935 |
| prefill_only/same_operator | hawq_signed | e2e_ms | 0.65 | 720.48 | hawq_signed_prefill_only_same_operator_r0.65 | 1 | 0 | 0 | 0.01139 |
| prefill_only/same_operator | hawq_signed | cycle_ms | 0.95 | 1053.16 | hawq_signed_prefill_only_same_operator_r0.95 | 3 | 0 | 0 | 0.00670 |
| prefill_only/same_operator | hawq_signed | cycle_ms | 0.80 | 886.87 | hawq_signed_prefill_only_same_operator_r0.8 | 2 | 0 | 0 | 0.00935 |
| prefill_only/same_operator | hawq_signed | cycle_ms | 0.65 | 720.59 | hawq_signed_prefill_only_same_operator_r0.65 | 1 | 0 | 0 | 0.01139 |

[完整36标签](latency_validation.csv) · [全部校验候选](calibration_candidates.csv) · [逐样本确认](confirmation_samples.csv) · [选择](selection.json) · [规则冻结](preregistered.json) · [原041结果](../../reviewer2/llama31_matched_allocator_run041/README.md) · [审计](provenance.json)
