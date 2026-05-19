## 2026-05-18 - INT4 SparseGPT fake checkpoint support
- 开发目的：新增与现有压缩方法平行的 `int4_unstructured_sparse` 和 `int4_semi_structured_sparse`，用于 full-Hessian SparseGPT + signed symmetric INT4 fake checkpoint 实验。
- 修改内容：新增 INT4 per-group fake quant；新增逐模块 full Hessian SparseGPT 压缩路径；接入 `CompressionConfig`、`SUPPORTED_METHODS`、prepare CLI、checkpoint CSV metadata；补充 README 和 opt-in 命令示例。
- 影响文件：`fake/compression/int4.py`、`fake/compression/sparsegpt.py`、`fake/compression/pipeline.py`、`scripts/prepare_compressed_model.py`、`fake/compression/checkpoint.py`、`README.md`、`scripts/slurm/all_model_test_commands.md`。
- 验证：`python3 -m py_compile fake/compression/*.py scripts/prepare_compressed_model.py scripts/eval_maxvit_dense_accuracy.py scripts/eval_dinov3_vit7b16_dense_accuracy.py` 通过；`bash -n scripts/slurm/prepare_compressed_models.sh scripts/slurm/eval_compressed_accuracy.sh` 通过；conda `wja-cospaq` 下完成 INT4 quant、pair-wise 2:4 mask、small Linear SparseGPT unstructured/structured smoke。
- 后续注意：本轮未实现 packed INT4 storage 或 INT4 sparse runtime kernel；新方法默认不加入批量 `METHODS`，需要显式 opt-in。
