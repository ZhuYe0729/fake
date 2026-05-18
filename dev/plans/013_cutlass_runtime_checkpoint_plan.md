# 013 CUTLASS Runtime-Packed Checkpoint Plan

## Summary
把当前“加载 dense/fake checkpoint 后现场转换成 CUTLASS module”的流程，升级为真正的 runtime-packed compressed checkpoint 流程。新增导出脚本先生成包含 CUTLASS packed buffers 的 checkpoint；后续 speed/accuracy 直接从该 packed checkpoint 构建 DINOv3 推理模型，不再依赖现场重新量化/剪枝。

## Key Changes
- 新增 `checkpoint_format="cutlass_runtime_packed_v1"`，保存非压缩模块权重、linear head、以及 280 个目标 Linear 的 CUTLASS packed buffers。
- 新增 `scripts/prepare_dinov3_cutlass_runtime_checkpoint.py`，支持 `--backend dense_nvfp4|sparse_nvfp4`。
- 新增 packed checkpoint loader，使用 meta-device skeleton 和 checkpoint 中的 packed tensors 构造 CUTLASS modules，不调用 `from_linear()`。
- 更新 CUTLASS dense/sparse speed 和 accuracy 脚本，增加 `--runtime-checkpoint`，CSV 记录 runtime checkpoint 信息。
- 新增 Slurm 导出入口，并让现有 CUTLASS speed/accuracy Slurm 支持 `RUNTIME_CHECKPOINT`。

## Test Plan
- `python -m py_compile` 覆盖新增 loader、导出脚本、更新后的 speed/accuracy 脚本。
- GPU 导出 dense 与 sparse runtime checkpoints。
- 验证 checkpoint 文件显著小于旧 26G float checkpoint，目标 Linear 不再保存 dense `.weight`。
- 用 `RUNTIME_CHECKPOINT=...` 跑 speed smoke、正式 speed、accuracy。

## Assumptions
- CUTLASS wrapper 的 `NVFP4Linear` / `SparseNVFP4Linear` state dict 已包含推理所需 packed buffers。
- sparse runtime checkpoint 首版优先复用已有 `nvfp4_semi_structured_sparse` 权重，避免改变剪枝口径。
