# 001 MaxViT Dense Eval Plan

## 目标
- 实现 `timm/maxvit_tiny_tf_224.in1k` dense 模型的 ImageNet 精度测试。
- 实现 dense 模型纯 forward 速度测试，并在结果中记录完整测试配置。
- 结果保存为 CSV，便于后续与 sparse/nvfp4 实验横向汇总。

## 实现范围
- 模型：本地路径 `/data/home/scxj523/run/wja/data/models/timm/maxvit_tiny_tf_224.in1k/`。
- 数据：本地 ImageNet val subset，使用 `val.csv + imagenet_val.zip`，不要求标准 ImageFolder 目录。
- 精度：top-1、top-5。
- 速度：随机输入，CUDA event 计时，不包含数据读取、解码、预处理。
- 环境：通过 Slurm 提交到 `gpu_5090`，激活 `wja-cospaq` conda 环境。

## 目录与接口
- `fake/models/maxvit.py`：加载 timm MaxViT 与本地权重。
- `fake/data/imagenet_zip.py`：从 zip 内按 csv 路径读取图片和标签。
- `fake/evaluation/accuracy.py`：ImageNet top-k 评估。
- `fake/evaluation/speed.py`：纯模型 forward benchmark。
- `scripts/eval_maxvit_dense_accuracy.py`：精度测试 CLI。
- `scripts/bench_maxvit_dense_speed.py`：速度测试 CLI。
- `scripts/slurm/*.sh`：超算提交脚本。
- `artifacts/results/maxvit_dense/*.csv`：实验结果。

## 默认配置
- dtype 默认 `auto`，保持模型原始 dtype。
- batch size 默认 128。
- ImageNet 输入尺寸 224，bicubic resize，center crop，ImageNet mean/std。
- speed benchmark 默认 warmup 50、iters 200。

## 验证
- 登录节点做 `compileall` 和 CLI 参数检查。
- 计算节点通过 sbatch 跑完整 accuracy/speed。
