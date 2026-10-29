# 局部特征粒度开发诊断（R3 LrT2 / R5 gkP9）

保持原模型/超参数/120标签及90 train/30 holdout划分，仅将3个新增8bit方法的局部特征由分桶均值恢复为已有逐模块测量。旧四种方法不变；没有读取confirmation或改变运行中策略。

| Model run | Scene | Split | Count | Original MAE | Module MAE | Original RMSE | Module RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 017 | prefill_only | train | 90 | 0.01887 | 0.01900 | 0.02792 | 0.02784 |
| 017 | prefill_only | holdout | 30 | 0.01544 | 0.01623 | 0.02203 | 0.02333 |
| 017 | prefill_only | new_methods_holdout | 12 | 0.00882 | 0.00892 | 0.01639 | 0.01705 |
| 017 | prefill_decode | train | 90 | 0.02294 | 0.02267 | 0.03160 | 0.03174 |
| 017 | prefill_decode | holdout | 30 | 0.03090 | 0.02980 | 0.04502 | 0.04325 |
| 017 | prefill_decode | new_methods_holdout | 12 | 0.02344 | 0.02309 | 0.03326 | 0.03178 |
| 018 | prefill_only | train | 90 | 0.01913 | 0.02001 | 0.02734 | 0.02636 |
| 018 | prefill_only | holdout | 30 | 0.01667 | 0.02503 | 0.02304 | 0.03376 |
| 018 | prefill_only | new_methods_holdout | 12 | 0.01122 | 0.01228 | 0.01370 | 0.01757 |
| 018 | prefill_decode | train | 90 | 0.01349 | 0.01350 | 0.01965 | 0.01963 |
| 018 | prefill_decode | holdout | 30 | 0.01520 | 0.01536 | 0.01884 | 0.01902 |
| 018 | prefill_decode | new_methods_holdout | 12 | 0.00626 | 0.00624 | 0.00878 | 0.00870 |

原holdout整体MAE仅Llama-3.1 P+D小幅改善，其他三组变差。该局部调整不支持整体改进结论，因此没有替换原方法或据此追加GPU配置。所有train/holdout/新方法holdout结果完整保留。

旧holdout是开发诊断，并非新独立测试；若后续提出新策略，需另固定新的validation/confirmation窗口与完整任务协议。

[CSV](results/feature_granularity.csv) · [设计](configs/design.json) · [审计](results/final_audit.json)
