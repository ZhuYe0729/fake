## 2026-05-17 - DINOv3 Four Over Six Fake Quant Paths
- 开发目的：为 DINOv3 ViT-7B/16 新增 NVFP4+unstructured 与 NVFP4+structured 的 4/6 fake 量化独立路径。
- 修改内容：新增 `four_over_six_mse` block scale 规则，扩展压缩 method 路由、checkpoint CSV 字段、DINOv3 专用 prepare/eval/bench 入口和 Slurm 脚本。
- 影响文件：`fake/compression/nvfp4.py`、`fake/compression/pipeline.py`、`fake/compression/checkpoint.py`、`scripts/prepare_compressed_model.py`、DINOv3 4/6 脚本、绘图脚本。
- 验证：`python3 -m py_compile` 通过；conda `wja-cospaq` 下 CPU fake quant 检查通过，确认 shape/dtype、`static_6` 默认不变、4/6 block 选择计数合法。
- 后续注意：当前只实现权重 fake 量化；激活量化 kernel 与 CUTLASS runtime 4/6 支持后续再接。

## 2026-05-18 - Add Activation Fake Quant
- 开发目的：让 DINOv3 4/6 fake-quant 路径同时覆盖 Linear 输入激活值。
- 修改内容：新增 activation NVFP4 fake quant 和 DINOv3 Linear wrapper；4/6 accuracy/speed 脚本默认在加载 checkpoint 后包裹 280 个目标 Linear，可用 `--no-activation-quant` 关闭。
- 影响文件：`fake/compression/nvfp4.py`、`fake/compression/activation.py`、`scripts/eval_dinov3_vit7b16_four_over_six_accuracy.py`、`scripts/bench_dinov3_vit7b16_four_over_six_speed.py`、对应 Slurm 脚本。
- 后续注意：当前仍是 PyTorch fake quant，预期会显著拖慢 speed；真实激活量化 kernel 后续再接。
