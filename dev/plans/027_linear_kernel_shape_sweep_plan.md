# 027 Linear Kernel Shape Sweep Benchmark Plan

## Summary
新增 standalone Linear shape sweep benchmark，用真实 kernel 路径评估 `dense_fp32`、`dense_bf16`、`sparse_bf16`、`dense_nvfp4`、`sparse_nvfp4` 在大量 `(m,n,k)` 下的速度和误差。`m` 表示 activation token 数，即 `batch * seq_len`；activation 为 `A[m,k]`，Linear weight 为 `W[n,k]`，输出为 `[m,n]`。

## Key Changes
- 新增 `scripts/bench_linear_kernel_shape_sweep.py`：
  - 参数：`--fixed-dim m|n|k`、`--output`、`--warmup`、`--iters`、`--seed`、`--fixed-values`、`--variable-values`。
  - 默认 fixed values：`1,16,64,256,4096,16384`。
  - 默认 variable values：`1,2,4,...,16384`。
  - 每次只跑一个 fixed dim，输出一个 CSV，便于提交 3 个 Slurm 任务并行运行。
- 每个 shape 生成同一组随机 `A[m,k]` 和 `W[n,k]`，然后按真实压缩口径准备各方法：
  - `dense_fp32`：`F.linear(A_fp32, W_fp32)`，作为误差 reference。
  - `dense_bf16`：PyTorch BF16 dense Linear。
  - `dense_nvfp4`：CUTLASS `NVFP4Linear`，权重离线 pack，forward 包含 activation 在线 NVFP4 quant/pack。
  - `sparse_bf16`：用 `hdiag = mean(A.float() ** 2, dim=0)` 调 `prune_dense_2_4(W, hdiag)`，再 pack 到真实 sparse BF16 runtime；forward 不做 activation quant。
  - `sparse_nvfp4`：用同一个 `hdiag` 调 `prune_nvfp4_pair_2_4(W, hdiag)`，再 pack 到真实 sparse NVFP4 runtime；forward 包含 activation 在线 NVFP4 quant/pack。
- CSV 输出 3 个主文件：
  - `artifacts/analysis/linear_kernel_shape_sweep/fixed_m.csv`
  - `artifacts/analysis/linear_kernel_shape_sweep/fixed_n.csv`
  - `artifacts/analysis/linear_kernel_shape_sweep/fixed_k.csv`
- CSV 每行是一组 `shape + method`，包含 shape、压缩耗时、forward latency、误差、speedup、kernel backend、sparsity、padding、设备和版本信息。
- 对 shape 不满足 kernel 对齐、OOM、runtime error 的方法写 `UNSUPPORTED` 或 `ERROR` 行后继续，不中断整批任务。

## Slurm Entry
- 新增 `scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`：
  - 使用 `gpu_5090`、`cuda/12.8`、`wja-cospaq`。
  - `FIXED_DIM=m|n|k` 控制任务。
  - 支持 `WARMUP`、`ITERS`、`OUTPUT`、`SEED`、`FIXED_VALUES`、`VARIABLE_VALUES` 覆盖。
  - 设置 CUTLASS wrapper build cache 环境变量，避免不同 kernel 扩展互相覆盖。

## Test Plan
- 登录节点静态检查：
  - `python -m py_compile scripts/bench_linear_kernel_shape_sweep.py`
  - `bash -n scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`
- GPU smoke：
  - `FIXED_DIM=m FIXED_VALUES=16 VARIABLE_VALUES=16,64 WARMUP=1 ITERS=3 sbatch scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`
  - 验证 CSV 中各方法都有 `OK/UNSUPPORTED/ERROR` 行。
  - 验证 `dense_fp32` 的 `mse=0`、`abs_max_error=0`。
- 正式运行：
  - `FIXED_DIM=m sbatch scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`
  - `FIXED_DIM=n sbatch scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`
  - `FIXED_DIM=k sbatch scripts/slurm/analysis/bench_linear_kernel_shape_sweep.sh`

## Assumptions
- 不测 bias，避免 bias add 干扰 GEMM/kernel 压缩收益。
- 压缩校准 activation 默认复用当前测试 shape 的同一份 `A`。
- 不考虑 four-over-six。
- `compress_ms` 单独记录；主速度比较使用 forward latency，不把离线权重压缩时间计入推理速度。
