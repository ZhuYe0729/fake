## 2026-07-01 - Batch Speed Sweep
- 开发目的：测试 MIRROR 在不同 batch size 下未压缩、uniform 压缩和混合压缩策略的端到端 forward 速度。
- 修改内容：
  - 新增 `run_batch_speed_sweep.py`，支持 dense default + AMP baseline、uniform 方法和混合候选策略的 batch sweep。
  - 新增 `summarize_batch_speed_sweep.py`，按 batch size 归一化到未压缩 AMP baseline，并为混合策略选择当前 batch 下实测最快候选。
  - 使用 GPU 0-5 并行完成 batch size 1/2/4/8/16/32 的速度验证。
  - 输出汇总 CSV、Markdown 和柱状图。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/scripts/run_batch_speed_sweep.py`
  - `artifacts/debug/030_mirror_global_pareto/scripts/summarize_batch_speed_sweep.py`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/batch_speed_sweep.csv`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_summary.csv`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_summary.md`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup.png`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/report/batch_speed_sweep_speedup.pdf`
  - `artifacts/debug/030_mirror_global_pareto/README.md`
- 后续注意：
  - 当前 sweep 为多 GPU 并行运行；不同 GPU/并发状态可能带来速度波动。若用于最终报告中的关键数值，建议对选中的 batch size 串行复测。

## 2026-07-01 - Batch 16 Sparse BF16 Recheck
- 开发目的：排查 batch sweep 中 `uniform_sparse_bf16` 在 batch=16 下比历史结果慢的问题。
- 修改内容：
  - 使用新 batch sweep 脚本在 GPU0 串行复测 batch=16，`uniform_sparse_bf16` 为 45.637753 ms。
  - 使用旧 `validate_pareto_speed.py` 原流程只复测 point 9，`uniform_sparse_bf16` 为 45.209 ms。
  - 查询 GPU 状态，GPU0 温度/时钟正常，无明显热降频或持续高负载。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/batch_speed_sweep_b16_serial_gpu0.csv`
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/recheck_uniform_sparse_bf16_validate_pareto_speed.csv`
- 后续注意：
  - 旧结果 `39.585003 ms` 与当前复测不一致，batch sweep 图中的 batch=16 数值不能直接和旧最终 Pareto 图混用；关键数字需要在同一脚本、同一串行环境下统一复测。

## 2026-07-01 - Batch 16 Sparse BF16 GPU1 Recheck
- 开发目的：排除 GPU0 单卡状态导致 batch=16 `uniform_sparse_bf16` 复测偏慢的可能。
- 修改内容：
  - 查询 GPU 状态，GPU1 为空闲低功耗状态。
  - 使用旧 `validate_pareto_speed.py` 流程在 `CUDA_VISIBLE_DEVICES=1` 上单独复测 point 9。
  - GPU1 复测结果为 44.309511 ms，接近 GPU0 当前复测的 45.208709 ms，仍明显慢于历史 39.585003 ms。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/speedaware_frontier/recheck_uniform_sparse_bf16_gpu1.csv`
- 后续注意：
  - 当前证据表明不是 GPU0 个体异常；历史 39.585 ms 更可能来自当时运行状态/环境差异或偶发偏快。后续速度图应避免混用历史速度和当前复测速度。

## 2026-07-01 - AMP Baseline Recheck
- 开发目的：确认未压缩 `dense_default + AMP` baseline 是否也发生变化，并排查 batch sweep baseline 口径。
- 修改内容：
  - 使用原 `amp_dense_default/scripts/validate_dense_amp_speed.py` 在 GPU1 上按 warmup=10、iters=50 复测。
  - 复测结果为 55.579517 ms，接近历史 56.528820 ms。
  - 发现新 `run_batch_speed_sweep.py` 中 baseline 使用 `torch.autocast(..., dtype=torch.bfloat16)`，而原 AMP baseline 使用 `torch.amp.autocast("cuda")` 默认 dtype；两者口径不一致。
- 影响文件：
  - `artifacts/debug/030_mirror_global_pareto/amp_dense_default_recheck_gpu1/speed/dense_default_use_amp_speed.csv`
- 后续注意：
  - batch sweep 需要修正 AMP baseline autocast 口径后重跑；当前 batch sweep 图不应用于汇报。
