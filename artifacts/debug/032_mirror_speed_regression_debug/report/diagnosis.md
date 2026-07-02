# MIRROR 速度异常排查结论

## 结论

这次看到的“各个压缩方法速度明显改变”主要不是压缩 runtime 或策略文件整体改坏，而是测速口径混用了不稳定/错误结果。

1. `dense_default_amp` 的 batch sweep 结果错误。
   - 原始 AMP dense baseline 使用 CUDA 默认 autocast dtype：`torch.float16`。
   - `run_batch_speed_sweep.py` 之前强制使用 `torch.bfloat16` autocast。
   - 因此 batch=16 baseline 从历史约 `56.53 ms` 被测成约 `62.47-62.79 ms`。
   - 已修复为 `torch.amp.autocast("cuda", enabled=True)`。

2. sparse BF16 runtime 本身没有发现退化。
   - 历史 `uniform_sparse_bf16`: `39.585003 ms`。
   - 在 `032` 中用原始 `validate_pareto_speed.py` 复测连续 sparse BF16 点时：
     - `uniform_sparse_bf16`: `39.588909 ms`。
   - 这与历史结果几乎完全一致。
   - 因此之前 batch sweep 中 `45.637753 ms` 的 `uniform_sparse_bf16` 是坏测量点，不应继续使用。

3. 部分 `batch_speed_sweep` 输出不稳定，尤其 uniform 全量替换点和 NVFP4 点。
   - 同一 batch=16 sweep 复跑时，`uniform_sparse_bf16` 有时回到 `39.47 ms`，有时又到 `43.10 ms`。
   - `uniform_sparse_nvfp4` 在不同 sweep 中出现 `93.13 ms`、`70.15 ms`、`82.26 ms`。
   - 这说明多 backend 混跑的一次性 sweep 对这些点不够可靠，不能作为最终图表数据来源。

## 已排除

- 策略文件不是原因：`uniform_sparse_bf16` 的历史/当前 policy 模块与 method 一致。
- sparse BF16 关键 runtime 文件没有 git diff。
- 当前 GPU 空闲状态未显示明显温度/占用异常。
- dense BF16 结果稳定：历史 `52.194736 ms`，复跑约 `52.18-52.29 ms`。

## 建议测速口径

后续 MIRROR 端到端速度应使用更隔离的测速协议：

- 每个策略单独进程运行，避免多 backend 在同一长进程中混跑。
- 使用 `validate_pareto_speed.py` 风格的单策略/少策略验证，而不是 `batch_speed_sweep.py` 的长 sweep 表直接出图。
- dense AMP baseline 必须使用默认 CUDA AMP，即 `torch.amp.autocast("cuda", enabled=True)`。
- uniform NVFP4 / sparse NVFP4 需要至少重复 3 次，使用 median 或 best-of-stable，而不是单次 mean。

## 关键结果文件

- 汇总表：`artifacts/debug/032_mirror_speed_regression_debug/results/speed_regression_summary.csv`
- sparse BF16 连续点复测：`artifacts/debug/032_mirror_speed_regression_debug/results/current_recheck_sparse_bf16_points_7_14_gpu1.csv`
- 修复前 batch sweep 复跑：`artifacts/debug/032_mirror_speed_regression_debug/results/current_batch_sweep_b16_gpu1_original_script.csv`
- 修复 AMP 后 batch sweep 复跑：`artifacts/debug/032_mirror_speed_regression_debug/results/current_batch_sweep_b16_gpu1_fixed_amp.csv`
