## 2026-05-08 - 初始 dense MaxViT 精度与速度测试
- 开发目的：支持 `timm/maxvit_tiny_tf_224.in1k` dense baseline 的 ImageNet 精度和纯 forward 速度测试。
- 修改内容：新增模型加载、zip/csv ImageNet 数据集、top-k 评估、CUDA event 速度测试、CSV 结果写入和 Slurm 提交脚本。
- 影响文件：`fake/`、`scripts/`、`README.md`、`dev/plans/001_maxvit_dense_eval_plan.md`。
- 后续注意：完整测试需要提交到 GPU 计算节点；登录节点无 GPU。
