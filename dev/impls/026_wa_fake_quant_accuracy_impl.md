## 2026-05-19 - WA fake quant 精度补测入口
- 开发目的：为缺少真实 kernel/runtime 的 `nvfp4_unstructured_sparse` 与 INT4 系列补充 WA fake quant 精度评测口径。
- 修改内容：新增 INT4 activation fake quant；activation wrapper 支持 `nvfp4`/`int4`；MaxViT 与 DINOv3 compressed accuracy 评测支持 activation fake quant；`eval_compressed_accuracy.sh` 增加 `WA_FAKE=1` 并写入 `accuracy_wa_fake.csv`；补充 README 与命令清单。
- 影响文件：`fake/compression/int4.py`、`fake/compression/activation.py`、`scripts/eval_maxvit_dense_accuracy.py`、`scripts/eval_dinov3_vit7b16_dense_accuracy.py`、`scripts/slurm/eval_compressed_accuracy.sh`、`scripts/slurm/all_model_test_commands.md`、`README.md`。
- 验证：`python3 -m py_compile fake/compression/*.py scripts/eval_maxvit_dense_accuracy.py scripts/eval_dinov3_vit7b16_dense_accuracy.py` 通过；`bash -n scripts/slurm/eval_compressed_accuracy.sh` 通过；`wja-cospaq` 环境下完成 INT4/NVFP4 Linear 与 INT4 Conv2d activation fake quant CPU smoke。
- 后续注意：`cutlass_sparse_nvfp4` 仍作为真实 WA structured NVFP4 runtime 主口径；本轮没有实现 INT4 kernel/runtime。

## 2026-05-19 - 修复 INT4 activation fake quant
- 开发目的：修复 MaxViT INT4 WA fake 精度坍塌，并避免不可整除 activation group 导致作业失败。
- 修改内容：INT4 activation fake quant 的动态 scale 保持 fp32，避免全零 group 的 `1e-12` clamp cast 到 fp16 后下溢为 0 并产生 NaN；activation wrapper 对最后一维不能被 group size 整除的模块直接跳过；MaxViT compressed eval 在默认 checkpoint 不存在时自动选择第一个 `METHOD_seed*/model.pt`。
- 影响文件：`fake/compression/int4.py`、`fake/compression/activation.py`、`scripts/slurm/eval_compressed_accuracy.sh`。
- 验证：`python3 -m py_compile fake/compression/*.py scripts/eval_maxvit_dense_accuracy.py scripts/eval_dinov3_vit7b16_dense_accuracy.py` 通过；`bash -n scripts/slurm/eval_compressed_accuracy.sh` 通过；`wja-cospaq` 下完成零 activation 不产生 NaN、不可整除 Linear 跳过、可整除 Linear 包装的 CPU smoke。
- 后续注意：已有 MaxViT INT4 WA fake 的 `0.100/0.500` 结果应视为无效，需要重跑；MaxViT tiny `nvfp4_unstructured_sparse` 可直接用 seed fallback 重跑。
