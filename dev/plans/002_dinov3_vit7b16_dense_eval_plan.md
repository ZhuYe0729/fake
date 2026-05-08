# 002 DINOv3 ViT-7B Dense Eval Plan

## 目标
- 实现 `facebook/dinov3-vit7b16-pretrain-lvd1689m` dense 模型的 ImageNet 精度测试。
- 使用本地 ImageNet linear head，和 DINOv3 README/hub classifier 的分类逻辑保持一致。
- 实现纯模型 forward 速度测试，结果保存为 CSV 并记录测试配置。

## 实现要点
- backbone 从本地 Hugging Face 目录加载：`AutoModel.from_pretrained(..., torch_dtype="auto", local_files_only=True)`。
- dense dtype 使用模型原始 dtype，不提供强制 dtype 转换。
- linear head 从本地 `.pth` 加载，输入为 `cls token + patch tokens mean`，跳过 4 个 register tokens。
- ImageNet 数据仍使用 `val.csv + imagenet_val.zip`，DINOv3 LVD transform 采用 README 建议的 256 resize 与 ImageNet mean/std。
- 默认 batch size 为 1，避免 7B 模型在 RTX 5090 上 OOM。

## 输出
- 精度：`artifacts/results/dinov3_vit7b16_dense/accuracy.csv`
- 速度：`artifacts/results/dinov3_vit7b16_dense/speed.csv`

## 验证
- 登录节点执行 `compileall` 和 shell 语法检查。
- 完整 accuracy/speed 通过 Slurm 提交到计算节点验证。
