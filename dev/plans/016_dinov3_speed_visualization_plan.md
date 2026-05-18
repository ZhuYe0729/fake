# DINOv3 Speed Visualization Plan

## 目标

分析 DINOv3 ViT-7B/16 在 dense、CUTLASS dense NVFP4、CUTLASS sparse NVFP4 storage 三种推理路径下，不同 batch size 的速度表现，并生成延迟与图片速度可视化结果。

## 数据来源

- `artifacts/results/dinov3_vit7b16_dense/speed.csv`
- `artifacts/results/dinov3_vit7b16_cutlass_nvfp4/speed.csv`
- `artifacts/results/dinov3_vit7b16_cutlass_sparse_nvfp4/speed_storage.csv`

## 实施步骤

1. 读取三份 speed CSV，按同一 batch size sweep 口径筛选最新有效记录。
2. 对比 `latency_mean_ms` 与 `images_per_sec` 两个指标。
3. 生成一张双子图 PNG，保存到 `artifacts/results/dinov3_speed_batchsize.png`。
4. 追加实现记录到 `dev/impls/016_dinov3_speed_visualization_impl.md`。

## 输出

- `artifacts/results/dinov3_speed_batchsize.png`
- `dev/impls/016_dinov3_speed_visualization_impl.md`
