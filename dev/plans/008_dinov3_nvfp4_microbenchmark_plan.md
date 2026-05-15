# 008 DINOv3 ViT-7B NVFP4 Microbenchmark Plan

## 目标

为 DINOv3 ViT-7B/16 增加与 MaxViT 类似的 FlashInfer NVFP4 microbenchmark：
- 加载 DINOv3 dense classifier。
- 将 selector 覆盖的 transformer Linear 替换为 `FlashInferNVFP4Linear`。
- 在典型输入尺寸下捕获真实 Linear 输入 shape。
- 逐层拆解 `layer_forward`、activation quant、`mm_fp4` GEMM、dense baseline 等耗时。
- 输出到 `artifacts/analysis/dinov3_vit7b16/nvfp4/microbench.csv`。

## 实现方案

1. 新增 `fake/models/dinov3_nvfp4.py`，提供 `load_dinov3_vit7b16_flashinfer_nvfp4_classifier`。
2. 新增 `scripts/bench_dinov3_vit7b16_nvfp4_micro.py`。
   - 默认输入尺寸：`3x128x128`、`3x256x256`、`3x384x384`。
   - 输入 H/W 校验为 16 的倍数。
   - 默认 batch size 为 1，warmup/iters 较 MaxViT 更小，避免 7B 模型测试时间过长。
3. 新增 `scripts/slurm/analysis/bench_dinov3_vit7b16_nvfp4_micro.sh`。
4. 补充 `scripts/README.md` 的 DINOv3/NVFP4 microbenchmark 条目。

## 注意事项

- DINOv3 7B 显存和运行时间明显高于 MaxViT，正式跑前建议先用 `MAX_LAYERS=3 WARMUP=2 ITERS=5` 冒烟。
- 该脚本只替换 compression selector 覆盖的 backbone transformer Linear，不替换分类 head。
