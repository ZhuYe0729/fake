# 031 MIRROR Hybrid CUTLASS Speed Plan

## Summary
为 MIRROR-DINOv3-Huge 增加手动 per-layer hybrid CUTLASS 速度实验，只看 forward speed，不做 accuracy、不做 hybrid checkpoint。实验目标是验证在 MIRROR 的 224 个 backbone transformer Linear 上，混合 sparse BF16 与 sparse NVFP4 是否能超过已有 best single runtime。

现有 MIRROR batch sweep 显示 single method 中 `semi_structured_sparse / Sparse BF16` 在各 batch 基本最强，例如 batch 16 `39.20 ms`、batch 32 `72.17 ms`，均明显优于 sparse NVFP4。因此本计划先实现多个手动 scheme，实测是否存在局部 NVFP4 替换能超过全 sparse BF16 的例外。

## Key Changes
- 新增 MIRROR hybrid loader/helper：
  - 遍历 `select_compressible_modules(model, "mirror")` 的 224 个 Linear。
  - 从原始 `nn.Linear` 直接构造目标 backend，避免重复替换。
  - MIRROR 的 `q/k/v` 后缀使用 `attention.{q,k,v}_proj.base_layer`。
  - sparse NVFP4 使用 `PaddedSparseNVFP4Linear(..., pad_multiple=32)`。
  - sparse BF16 使用 `PaddedSparseBF16Linear(..., pad_multiple=8)`。
- 固定支持 4 个手动方案：
  - `dino_b16_like`: `q/k/v/o/gate/up -> sparse_nvfp4`，`down -> sparse_bf16`。
  - `dino_b32_like`: `gate/up -> sparse_nvfp4`，`q/k/v/o/down -> sparse_bf16`。
  - `attn_nvfp4_mlp_bf16`: `q/k/v/o -> sparse_nvfp4`，`gate/up/down -> sparse_bf16`。
  - `attn_bf16_mlp_nvfp4`: `q/k/v/o -> sparse_bf16`，`gate/up/down -> sparse_nvfp4`。
- 新增 speed benchmark 脚本：
  - `scripts/bench_mirror_cutlass_hybrid_speed.py`
  - CSV 写入 `artifacts/results/mirror_cutlass_hybrid/speed.csv`。
  - 记录 `method=hybrid_cutlass`、`hybrid_scheme`、两类模块数、替换/跳过数量、latency 和 throughput。
- 新增 Slurm 脚本：
  - `scripts/slurm/bench_mirror_cutlass_hybrid_speed.sh`
  - 默认跑 batch `1 2 4 8 16 32` 和上述 4 个 scheme。
  - 复用 MIRROR 环境设置、offline HF 设置和 CUTLASS extension build dir。

## Test Plan
- 静态检查：
  - `python -m py_compile` 覆盖新增 Python 文件。
  - `bash -n` 覆盖新增 Slurm 脚本。
- GPU 节点速度验证：
  - 冒烟：`WARMUP=2 ITERS=5 BATCH_SIZES="16 32" HYBRID_SCHEMES="dino_b32_like" sbatch scripts/slurm/bench_mirror_cutlass_hybrid_speed.sh`
  - 正式：`WARMUP=10 ITERS=50 sbatch scripts/slurm/bench_mirror_cutlass_hybrid_speed.sh`
  - 重点比较 `artifacts/results/mirror_compressed/speed_batch_sweep_summary.csv` 中每个 batch 的 Sparse BF16 best single。

## Assumptions
- 本轮只验证速度，不做 MIRROR accuracy。
- 输入沿用 MIRROR speed 设置：`3x224x224`。
- 只压缩 MIRROR backbone transformer 224 个 Linear；memory bank、detector、patch embedding、norm 和 LoRA adapter 小矩阵保持原状。
