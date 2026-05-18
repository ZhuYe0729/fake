# 012 DINOv3 CUTLASS Sparse NVFP4 Inference Plan

## Summary
为 DINOv3 ViT-7B/16 新增独立 CUTLASS structured sparse NVFP4 推理路径，复用 `fake/kernels/cutlass/cutlass_wrapper` 中的 `SparseNVFP4Linear`，替换 backbone transformer 内 280 个 compressible `Linear`，并保留现有 dense NVFP4、FlashInfer 和 fake-quant checkpoint 路径。

## Key Changes
- 新增 `fake/kernels/cutlass_sparse_nvfp4.py` adapter，导入 sparse wrapper API，提供 config、replacement report、模块替换和计数。
- adapter 用 padding wrapper 处理 DINOv3 默认 token 数不是 32 倍数的问题：每个 Linear forward 内部补齐 flattened token 行数，kernel 输出后切回原 shape。
- 新增 `fake/models/dinov3_cutlass_sparse_nvfp4.py` loader，直接 bf16 加载 DINOv3 classifier，并将目标 Linear 替换为 sparse NVFP4。
- 新增 speed / accuracy 脚本和 Slurm 入口，输出到 `artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/`。
- 脚本支持可选 checkpoint；默认转换时 `prune=True`，若加载已有结构化稀疏 checkpoint 可设置 `NO_PRUNE=1` 走 strict conversion。
- 更新 `scripts/README.md`，说明 sparse NVFP4 为独立真实 kernel 路径。

## Test Plan
- `python -m py_compile fake/kernels/cutlass_sparse_nvfp4.py fake/models/dinov3_cutlass_sparse_nvfp4.py scripts/bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.py scripts/eval_dinov3_vit7b16_cutlass_sparse_nvfp4_accuracy.py`
- 登录节点导入检查，确认 adapter 能找到 `SparseNVFP4Linear` 且 shape predicate 不触发 CUDA extension build。
- GPU 节点先跑 wrapper correctness：`sbatch fake/kernels/cutlass/cutlass_wrapper/scripts/test_sparse_nvfp4_correctness_5090.sh`。
- GPU 节点 smoke：`WARMUP=1 ITERS=2 sbatch scripts/slurm/bench_dinov3_vit7b16_cutlass_sparse_nvfp4_speed.sh`。
- 正式运行 speed 与 accuracy，检查 replaced count 为 280、skipped count 为 0。

## Assumptions
- sparse wrapper 使用 pairwise 4:8 structured sparsity；默认 `prune=True` 在转换时按 magnitude 做结构化剪枝。
- 使用已有 `nvfp4_semi_structured_sparse` checkpoint 时，checkpoint 权重已满足 pairwise 4:8 稀疏模式，适合配合 `NO_PRUNE=1`。
- CUTLASS sparse NVFP4 运行 dtype 固定为 bf16，权重量化/pack 发生在 loader 阶段。
- Padding 只用于满足 kernel token 对齐，不改变模型可见输出 shape。
