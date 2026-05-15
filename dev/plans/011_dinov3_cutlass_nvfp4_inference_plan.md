# 011 DINOv3 CUTLASS NVFP4 Inference Plan

## Summary
为 DINOv3 ViT-7B/16 新增独立的 CUTLASS dense NVFP4 推理路径，用 `fake/kernels/cutlass/cutlass_wrapper` 中的 `NVFP4Linear` 替换 backbone transformer 内 280 个 compressible `Linear`，保留现有 dense、checkpoint fake-quant、FlashInfer NVFP4 路径不变。

## Key Changes
- 新增 `fake/kernels/cutlass_nvfp4.py` adapter，从 nested symlink package 导入 CUTLASS wrapper API，提供 config、replacement report、Linear 替换与模块计数。
- 新增 `fake/models/dinov3_cutlass_nvfp4.py` loader，加载 dense classifier 后转为 bf16，并将 DINOv3 backbone transformer projection Linear 替换为 CUTLASS `NVFP4Linear`。
- 新增独立 speed 与 accuracy 脚本，输出到 `artifacts/results/dinov3_vit7b16_cutlass_nvfp4/`，CSV 记录 backend、替换计数、runtime dtype 和运行配置。
- 新增两个 Slurm 入口，加载 `cuda/12.8`、激活 `wja-cospaq`、设置 offline env，并支持通过环境变量覆盖常用参数。
- 更新 `scripts/README.md`，说明 CUTLASS 路径与现有 FlashInfer 路径并存。

## Test Plan
- `python -m py_compile fake/kernels/cutlass_nvfp4.py fake/models/dinov3_cutlass_nvfp4.py scripts/bench_dinov3_vit7b16_cutlass_nvfp4_speed.py scripts/eval_dinov3_vit7b16_cutlass_nvfp4_accuracy.py`
- 登录节点导入检查，确认 adapter 能找到 nested CUTLASS wrapper package 且不强制 build CUDA extension。
- GPU 节点 smoke：`WARMUP=1 ITERS=2 sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_nvfp4_speed.sh`，确认 replaced count 为 280、skipped count 为 0、CSV 有有效 latency。
- 正式运行 speed 与 accuracy Slurm 脚本，并与 dense baseline / FlashInfer 结果横向比较。

## Assumptions
- CUTLASS wrapper 已在 RTX 5090 / SM120 节点通过基础 correctness；主仓库只负责接入和 end-to-end 验证。
- DINOv3 CUTLASS NVFP4 运行 dtype 固定为 bf16。
- 权重量化/pack 成本发生在 loader 阶段，不计入每次 forward latency。
- 现有 compressed checkpoint fake-quant 路径不改变。
