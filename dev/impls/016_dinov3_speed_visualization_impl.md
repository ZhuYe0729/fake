## 2026-05-16 - DINOv3 speed batch-size visualization

- 开发目的：对比 DINOv3 ViT-7B/16 的 dense、CUTLASS dense NVFP4、CUTLASS sparse NVFP4 storage 在不同 batch size 下的延迟与图片速度。
- 修改内容：
  - 新增 `scripts/plot_dinov3_speed_batchsize.py`，读取三条 speed CSV，并筛选 `warmup=5,iters=20` 的最新 batch sweep 记录。
  - 生成双子图：左侧为 mean latency，右侧为 images/s throughput。
  - 输出 `artifacts/results/dinov3_speed_batchsize.png`。
- 影响文件：
  - `scripts/plot_dinov3_speed_batchsize.py`
  - `artifacts/results/dinov3_speed_batchsize.png`
  - `dev/plans/016_dinov3_speed_visualization_plan.md`
  - `dev/impls/016_dinov3_speed_visualization_impl.md`
- 结果摘要：
  - dense fp32：batch 1 延迟 106.144 ms，最佳吞吐 batch 128 为 14.967 img/s。
  - CUTLASS dense NVFP4：batch 1 延迟 38.093 ms，最佳吞吐 batch 8 为 81.746 img/s。
  - CUTLASS sparse NVFP4：batch 1 延迟 39.218 ms，最佳吞吐 batch 8 为 87.387 img/s。
- 后续注意：sparse NVFP4 在 batch 8 的吞吐最高，但 batch 1 延迟略慢于 dense NVFP4；如需要论文/报告图，可再导出 PDF 或补充 speedup ratio 图。
