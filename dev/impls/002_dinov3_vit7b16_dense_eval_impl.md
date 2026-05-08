## 2026-05-08 - 初始 DINOv3 ViT-7B dense 精度与速度测试
- 开发目的：支持 `dinov3-vit7b16-pretrain-lvd1689m` dense baseline 的 ImageNet 精度和纯 forward 速度测试。
- 修改内容：新增 DINOv3 Transformers backbone + ImageNet linear head 包装、DINOv3 transform、精度/速度 CLI 和 Slurm 脚本。
- 影响文件：`fake/models/dinov3.py`、`fake/data/dinov3_transforms.py`、`fake/data/imagenet_zip.py`、`scripts/`、`README.md`、`dev/plans/002_dinov3_vit7b16_dense_eval_plan.md`。
- 后续注意：7B 模型默认 batch size 为 1，完整测试需要提交到 GPU 计算节点。
