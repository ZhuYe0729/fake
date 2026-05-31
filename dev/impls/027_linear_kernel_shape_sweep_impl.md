## 2026-05-25 - Linear kernel shape sweep benchmark
- 开发目的：新增按 `(m,n,k)` sweep 的真实 Linear kernel benchmark，用于比较 dense、BF16 sparse、NVFP4、sparse NVFP4 在不同权重和 activation shape 下的速度与误差。
- 修改内容：新增 benchmark 脚本和 Slurm 入口；每个 shape 复用同一份 `A[m,k]`/`W[n,k]`，用 `dense_fp32` 作为误差 reference，sparse 方法按 Hessian diag pruning 后再进入真实 runtime pack。
- 影响文件：`scripts/bench_linear_kernel_shape_sweep.py`、`scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`、`dev/plans/027_linear_kernel_shape_sweep_plan.md`。
- 验证结果：`python -m py_compile scripts/bench_linear_kernel_shape_sweep.py`、`bash -n scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`、`python scripts/bench_linear_kernel_shape_sweep.py --help` 通过。
- 后续注意：正式运行应分别提交 `FIXED_DIM=m/n/k` 三个任务；大 shape 可能 OOM 或耗时较长，脚本会按 method 写 `UNSUPPORTED`/`ERROR` 后继续；GPU smoke 尚未提交。

## 2026-05-25 - Submit full shape sweep jobs
- 开发目的：提交完整 fixed-dim shape sweep benchmark。
- 修改内容：使用 `WARMUP=5`、`ITERS=10` 分别提交 `FIXED_DIM=m/n/k` 三个 Slurm 任务。
- 影响文件：预计输出 `artifacts/analysis/linear_kernel_shape_sweep/fixed_m.csv`、`fixed_n.csv`、`fixed_k.csv`。
- 后续注意：Slurm job id 为 `469787`、`469789`、`469788`。

## 2026-05-25 - Resume support after quota stop
- 开发目的：清理空间后续跑未完成的 shape sweep，避免重复写已经完成的 shape/method。
- 修改内容：`bench_linear_kernel_shape_sweep.py` 新增 `--resume`，读取已有 CSV 并跳过已存在的 `shape_index + method`；Slurm 入口默认 `RESUME=1`。
- 影响文件：`scripts/bench_linear_kernel_shape_sweep.py`、`scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`。
- 验证结果：`python -m py_compile scripts/bench_linear_kernel_shape_sweep.py`、`bash -n scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh` 通过；现有 CSV 仅各有一个 partial shape 需要补齐后继续。

## 2026-05-25 - Full sweep completed and analyzed
- 开发目的：确认续跑任务完成，并刷新完整结果分析。
- 修改内容：提交续跑 job `470902`、`470903`、`470904`；三份 fixed-dim CSV 均补齐到 6750 行、1350 个 shape；更新结果目录内 `summary.md`、汇总 CSV 和 PNG。
- 影响文件：`artifacts/analysis/linear_kernel_shape_sweep/*`、`dev/impls/027_linear_kernel_shape_sweep_impl.md`。
- 后续注意：最终结果中仍包含 expected unsupported rows 和 sparse NVFP4 runtime error rows，需结合 `top_error_messages.csv` 过滤分析。

## 2026-05-25 - Best-method matrix plots
- 开发目的：按 fixed-dim 方式可视化每个 shape 下速度最快的方法。
- 修改内容：新增 Pillow 绘图脚本，生成 `fixed_m`、`fixed_n`、`fixed_k` 三张 best-method 矩阵图；每个图包含 6 个固定值子图，格子颜色和编号均表示 fastest OK method。
- 影响文件：`artifacts/analysis/linear_kernel_shape_sweep/plot_best_method_matrices.py`、`best_method_matrix_fixed_m.png`、`best_method_matrix_fixed_n.png`、`best_method_matrix_fixed_k.png`。
- 后续注意：编号映射为 `0=dense_fp32`、`1=dense_bf16`、`2=sparse_bf16`、`3=dense_nvfp4`、`4=sparse_nvfp4`，`-` 表示没有 OK method。
