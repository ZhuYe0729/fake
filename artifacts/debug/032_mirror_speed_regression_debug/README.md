# MIRROR Speed Regression Debug

This directory isolates the investigation of MIRROR speed discrepancies across historical and current measurements.

## 结论摘要

- `dense_default_amp` 速度变化来自 `run_batch_speed_sweep.py` 的 AMP 口径 bug：脚本曾强制 BF16 autocast，而原始 baseline 使用 CUDA 默认 AMP (`float16`)。
- sparse BF16 runtime 本身没有退化：`uniform_sparse_bf16` 用原始验证脚本复测为 `39.588909 ms`，与历史 `39.585003 ms` 对齐。
- 之前的 batch sweep 文件包含坏测量点，不应继续用于最终图表；后续端到端速度建议每个策略独立进程复测。

## 关键文件

- `report/diagnosis.md`
- `results/speed_regression_summary.csv`
- `results/current_recheck_sparse_bf16_points_7_14_gpu1.csv`
- `results/current_batch_sweep_b16_gpu1_original_script.csv`
- `results/current_batch_sweep_b16_gpu1_fixed_amp.csv`
- `corrected_batch_speed_sweep_by_batch/report/batch_speed_sweep_speedup.png`
- `../030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_corrected_by_batch.png`
- `corrected_batch_speed_sweep_steady_sparse_first_w100_i50/report/batch_speed_sweep_speedup.png`
- `../030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_steady_sparse_first_w100_i50.png`
- `../030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_steady_sparse_first_w100_i50_gray_red.png`
- `../030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_compact_8_16_32_gray_red.png`
- `../030_mirror_global_pareto/speedaware_frontier/report/batch_strategy_cards_8_16_32.png`
- `../030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup_compact_8_16_32_shape_stable_gray_red.png`
- `../030_mirror_global_pareto/speedaware_frontier/report/batch_strategy_cards_8_16_32_shape_stable.png`
