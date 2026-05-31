# 030 DINOv3 Hybrid CUTLASS Speed Demo Plan

## Summary
实现一个最小可验证的 DINOv3-7B 手动混合压缩速度实验，只测 `batch=16` 和 `batch=32`，不做 planner、不做 accuracy、不做 hybrid checkpoint。加载方式采用在线替换：先加载 dense BF16 DINOv3，再按模块名后缀把不同 Linear 替换成 sparse NVFP4 或 sparse BF16。

目标输出：`artifacts/results/dinov3_vit7b16_cutlass_hybrid/speed.csv`，用于和现有单一 sparse BF16 / sparse NVFP4 结果对比。

## Key Changes
- 新增 DINOv3 hybrid loader/helper：
  - `b16_manual`: `q/k/v/o/gate/up -> sparse_nvfp4`，`down -> sparse_bf16`。
  - `b32_manual`: `q/k/v/o/down -> sparse_bf16`，`gate/up -> sparse_nvfp4`。
  - 遍历 `select_compressible_modules(model, "dinov3_vit7b16")` 的 280 个 Linear，并从原始 `nn.Linear` 直接构造目标 backend。
- 新增 speed benchmark 脚本：
  - 参数包含 `--hybrid-scheme {b16_manual,b32_manual}`、`--batch-size`、`--input-size`、`--warmup`、`--iters`、`--output`。
  - CSV 记录 `method=hybrid_cutlass`、`hybrid_scheme`、两类模块数量、替换/跳过数量、latency 和吞吐。
- 新增 Slurm 脚本：
  - 默认只跑 `batch=16/b16_manual` 和 `batch=32/b32_manual`。
  - 复用现有 CUDA 12.8、`wja-cospaq`、offline HF 环境和 CUTLASS extension build dir 设置。

## Test Plan
- 静态检查：
  - `python3 -m py_compile` 覆盖新增 Python 文件。
  - `bash -n` 覆盖新增 Slurm 脚本。
- GPU 节点速度验证：
  - 先用 `WARMUP=2 ITERS=5` 冒烟测试。
  - 正式跑 `WARMUP=5 ITERS=20` 或沿用现有 batch sweep 参数。
  - 验证 CSV 中 `batch=16` 为 sparse NVFP4 240 个、sparse BF16 40 个；`batch=32` 为 sparse NVFP4 80 个、sparse BF16 200 个；总替换 280、跳过 0。

## Assumptions
- 本轮只验证速度，不做 ImageNet accuracy。
- 不新增 hybrid checkpoint；在线替换的构建/压缩时间不计入 forward benchmark。
- 输入固定沿用现有 DINOv3 speed 设置：`3x256x256`。
- 只压缩 backbone transformer 280 个 Linear；`linear_head`、patch embedding、norm 和非 Linear 模块保持现状。
