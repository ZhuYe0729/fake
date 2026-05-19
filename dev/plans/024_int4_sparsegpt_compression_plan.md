# 024 INT4 SparseGPT Compression Plan

## Summary
新增两个与现有压缩方法平行的 fake-checkpoint 方法：`int4_unstructured_sparse` 和 `int4_semi_structured_sparse`。它们接入当前 `prepare_compressed_model.py`、accuracy、metadata 体系，不改变现有 NVFP4、sparse、CUTLASS 路径，也不默认加入批量 Slurm `METHODS`。

## Key Changes
- 新增 signed symmetric per-group INT4 fake quant：unstructured 默认 group size 32，semi-structured 默认 group size 64，每组保存 fp16 scale，无 global scale 和 zero point。
- 新增 `int4_*` 专用 full SparseGPT 路径：逐模块收集 full Hessian，使用 block-wise Cholesky inverse error compensation，默认 `blocksize=128`、`percdamp=0.01`。
- `int4_unstructured_sparse` 使用 50% unstructured SparseGPT mask；`int4_semi_structured_sparse` 使用 pair-wise 2:4 over 8 columns，列数不能被 8 整除的模块跳过并记录 metadata。
- 扩展 `CompressionConfig`、prepare CLI、checkpoint CSV 字段、README 和命令文档；旧方法默认值与行为保持不变。

## Tests
- `python -m py_compile fake/compression/*.py scripts/prepare_compressed_model.py scripts/eval_*_accuracy.py`
- `bash -n scripts/slurm/prepare_compressed_models.sh scripts/slurm/eval_compressed_accuracy.sh`
- 小矩阵 smoke 验证 INT4 group quant 和 pair-wise 2:4 mask。
- GPU 节点 opt-in 生成 MaxViT tiny 两个 int4 checkpoint 后跑 accuracy。

## Assumptions
- 本轮不实现 packed INT4 storage 或 INT4 sparse runtime kernel。
- 新方法不加入 Slurm 默认 `METHODS`，需要显式 opt-in。
- Full SparseGPT 只用于 `int4_*`，现有方法继续使用原来的 diag Hessian pipeline。
