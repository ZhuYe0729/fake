# WA Fake Quant 补测实现计划

- 开发目的：新增 WA fake quant 精度评测，用于补测没有真实 kernel/runtime 覆盖的 `nvfp4_unstructured_sparse`、`int4`、`int4_semi_structured_sparse`、`int4_unstructured_sparse`。
- 修改内容：补充 INT4 activation fake quant；扩展通用 activation wrapper 支持 `nvfp4`/`int4`；为 MaxViT 与 DINOv3 compressed accuracy 评测增加 WA fake 参数和 `accuracy_wa_fake.csv` 输出口径；更新 Slurm 命令与文档。
- 影响文件：`fake/compression/int4.py`、`fake/compression/activation.py`、`scripts/eval_maxvit_dense_accuracy.py`、`scripts/eval_dinov3_vit7b16_dense_accuracy.py`、`scripts/slurm/eval_compressed_accuracy.sh`、`scripts/slurm/all_model_test_commands.md`、`README.md`。
- 后续注意：`cutlass_sparse_nvfp4` 已覆盖真实 WA structured NVFP4 runtime；本计划不实现 INT4 kernel/runtime，只补 fake quant 精度。
