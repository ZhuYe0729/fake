# 015 MaxViT CUTLASS NVFP4 End-to-End Plan

## Summary
为 MaxViT 补齐类似 DINOv3 的端到端 dense、CUTLASS dense NVFP4、CUTLASS sparse NVFP4 推理链路。保留现有 FlashInfer NVFP4 路径，新增独立 CUTLASS 路径用于真实 kernel 的 speed/accuracy 验证。

## Key Changes
- 新增 MaxViT CUTLASS dense NVFP4 loader，加载 dense bf16 后替换 compressible Linear 为 `NVFP4Linear`。
- 新增 MaxViT CUTLASS sparse NVFP4 loader，加载 dense bf16 后替换 compressible Linear 为 `PaddedSparseNVFP4Linear`，默认 conversion 时 prune。
- 新增 MaxViT CUTLASS dense/sparse speed 和 ImageNet accuracy 脚本。
- 新增对应 Slurm 入口，支持 variant、batch/input/warmup/iters/output 等环境变量。
- 更新 `scripts/README.md`，说明 MaxViT dense、FlashInfer NVFP4、CUTLASS dense NVFP4、CUTLASS sparse NVFP4 的关系。

## Test Plan
- `python -m py_compile` 覆盖新增 loader 和 speed/accuracy 脚本。
- `bash -n` 覆盖新增 Slurm 脚本。
- GPU smoke：tiny variant 先跑 speed smoke，确认 replaced/skipped counts 和 CSV 写入。
- GPU accuracy：tiny variant 跑 ImageNet accuracy，与 dense 和已有 compressed/fake 结果比较。

## Assumptions
- MaxViT CUTLASS 首版只替换 `select_compressible_modules(model, "maxvit")` 中的 Linear；MBConv pointwise Conv2d 仍保持 dense，因为当前 CUTLASS wrapper 是 Linear/GEMM 封装。
- sparse NVFP4 首版默认从 dense 权重 conversion 时 prune；如提供 fake sparse checkpoint，可通过 `--checkpoint --no-prune` 复用已有剪枝口径。
