## 2026-05-08 - 初始 NVFP4 与剪枝 pipeline
- 开发目的：实现 MaxViT 和 DINOv3 ViT-7B 的压缩 checkpoint 生成与复用评估入口。
- 修改内容：新增 NVFP4 fake quant、Hessian diagonal 校准、unstructured/2:4/NVFP4 pair-wise 2:4 剪枝、模块选择、checkpoint 加载、压缩 CLI 和 Slurm 脚本。
- 影响文件：`fake/compression/`、`scripts/prepare_compressed_model.py`、现有 accuracy/speed 脚本、`scripts/slurm/`、`README.md`、`dev/plans/003_compression_pipeline_plan.md`。
- 后续注意：第一版保存 dequantized fake-quant checkpoint；`masks.pt`/`scales.pt` 默认保存 metadata-only，完整 mask/scale 张量需显式开启。
