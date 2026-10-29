# Qwen3：完整候选空间质量修正快速实验

144个模块独立选择prefill/decode，8×8动作；分桶仅是共享校准特征，不约束模块同选。旧/新使用相同LUT及精确MILP，允许signed质量成本。两者均以BF16预测为0，旧bias移除的影响单独记录。

复用训练期标签：130个train、44个policy holdout，前64个相同训练窗口。016 eager与029 Graph来源以各自BF16配对；BF16最大逐窗口差1.91e-07，不宣称所有混合策略协议严格等价。

旧锚定代理holdout MAE=0.019208，signed残差=0.015779；ridge=0.1由固定policy holdout选择，这不是最终新窗口误差。本轮旧代理冻结，残差使用130个训练标签；新增校准标签与结构改变的作用未隔离，不能把MAE差异全部归因于新结构。

校正代理四个自动提案中，0个通过新validation的.01保护值，2个通过.04。完整逐提案预测/实测偏差见proposal_diagnostics.csv，不能用整体holdout MAE代替优化器实际提案上的准确度。

128新validation决定质量可行性，同卡calibration E2E决定选点；冻结后64新confirmation。所有新质量/速度输入排除原训练及028/029/030窗口。保护值同mean+2SE+.005。主目标是原论文单批E2E，包含prefill→decode实际切换，批间恢复单列，不强加到主指标。

| 模型 | ε | 策略 | Val guard | Confirm mean | Confirm guard | Guard pass | E2E ms | ×BF16 | ×INT8 |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| old_anchored | 0.01 | group_3ee5 | -0.00299 | -0.00026 | 0.02135 | False | 1128.77 | 1.772 | 1.084 |
| old_anchored | 0.04 | old_anchored_0.04 | 0.01815 | -0.00559 | 0.02299 | True | 976.87 | 2.047 | 1.253 |
| signed_residual | 0.01 | group_3ee5 | -0.00299 | -0.00026 | 0.02135 | False | 1128.77 | 1.772 | 1.084 |
| signed_residual | 0.04 | signed_residual_0.04 | 0.03980 | 0.01424 | 0.04730 | False | 959.31 | 2.085 | 1.276 |

各模型只使用自身四目标提案和共同anchor。030的90-W4仅是共同incumbent，不能将复用它算作本轮完整空间新发现。所有失败提案保留；不按确认或正式速度重选。本轮没有完整生成/MC，NLL可行不等于所有下游任务不退化，也未新增HAWQ对照。

接入时首次复制的checkpoint helper遗漏canonical稀疏参数，导出报错后补齐旧016相同SparseGPT状态；未更改模型/提案/数据/预算。成功任务保留，全部初次失败和源码版本保留，恢复记录及canonical SHA另存results/sparse_export_recovery.json。

两个.04选点差异为15个模块；新减旧配对确认ΔNLL=0.019830，具体模块与方法见selected_policy_diff.json。此诊断只描述已冻结结果，不回填训练或调整预算。

3轮GPU0/NUMA0，3warmup+5timed、首尾BF16、native decode Graph及6GiB KV。原始轮次与恢复开销保留。

[所有validation](results/validation.csv) · [提案预测与实测](results/proposal_diagnostics.csv) · [共同12点的预测](results/common_pool_predictions.csv) · [固定前缀窗口敏感性](results/window_sensitivity.csv) · [选择与确认](results/selected_results.csv) · [三轮速度](results/formal_campaigns.csv) · [各策略精度分配](results/policy_counts.json) · [模型拟合](results/model_fit.json)
