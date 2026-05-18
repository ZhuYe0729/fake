# 014 CUTLASS Storage Checkpoint Plan

## Summary
在 013 runtime-packed checkpoint 之外新增更省磁盘的 storage checkpoint 层。storage checkpoint 保存稀疏 NVFP4 的 compact pairwise 4:8 FP4 表示；需要推理时再转换为 CUTLASS runtime checkpoint 的 `sparse_weight + metadata` kernel-ready buffers。

## Key Changes
- 新增 `checkpoint_format="cutlass_storage_packed_v1"`，首版覆盖 DINOv3 sparse NVFP4。
- storage checkpoint 保存非压缩模块权重和 280 个目标 Linear 的 `storage_values / pair_mask / weight_scale / weight_global_scale / bias`。
- 新增 storage 导出脚本，从已有 `nvfp4_semi_structured_sparse` fake checkpoint 量化一次并保存 compact storage，不保存 CUTLASS runtime `sparse_weight/metadata`。
- 扩展 runtime prepare 脚本，支持从 storage checkpoint 转成 runtime checkpoint，转换时只做 FP4 pair unpack + CUTLASS sparse pack，不重新 prune/quant。
- 更新 Slurm 和 README，明确 storage checkpoint 用于磁盘保存，并支持加载时转换成 runtime buffers 后直接 kernel 推理；runtime checkpoint 作为可选缓存格式保留。

## Test Plan
- `python -m py_compile` 覆盖新增 storage 模块/脚本和更新后的 runtime prepare 脚本。
- `bash -n` 覆盖新增/更新 Slurm 脚本。
- GPU 导出 sparse storage checkpoint，确认文件大小小于 sparse runtime checkpoint。
- 直接用 `STORAGE_CHECKPOINT=...` 跑 speed/accuracy，结果应与 013 sparse runtime 基本一致；可选再从 storage checkpoint 转 runtime checkpoint 做缓存验证。

## Assumptions
- CUTLASS wrapper 的 `pack_sparse_nvfp4_a_from_dense()` 可从 dense-style packed FP4 重建 runtime sparse buffers。
- storage checkpoint 不直接存 kernel-ready buffers；加载时需要在内存中转换成 runtime buffers 后进入现有推理路径。
