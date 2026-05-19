## 2026-05-18 - End-to-End Speedup PPT Figure
- 开发目的：生成用于专业 PPT 的端到端速度结果页，展示真实 CUTLASS 推理路径相对 dense baseline 的 throughput speedup。
- 修改内容：新增 `scripts/plot_end_to_end_speedup_summary.py`，读取 MaxViT tiny/small/base/large 和 DINOv3 的 speed CSV，输出 grouped horizontal bar chart 和配套 CSV。
- 影响文件：`scripts/plot_end_to_end_speedup_summary.py`、`artifacts/results/end_to_end_speedup_summary.*`。
- 验证：`python3 -m py_compile` 通过；在 `wja-cospaq` 环境下成功生成 PNG/PDF/CSV，并预览确认标题、图例和标签无重叠。
- 后续注意：当前速度页只展示真实部署路径 `Dense NVFP4`、`4:8 Sparse BF16`、`4:8 Sparse NVFP4`；Rescale/four-over-six 暂无真实 packed kernel 速度，因此不放入速度图。

## 2026-05-18 - Add DINOv3 Batch Sweep Panels
- 开发目的：优化速度页视觉表达，避免右侧单个 DINOv3 横条过重，并补充 DINOv3 batch-size sweep 速度结果。
- 修改内容：左侧 MaxViT bar 标签改为只显示 speedup；右侧改为两个 DINOv3 小图，分别展示 batch size sweep 的相对 dense speedup、以及 Sparse NVFP4 相对 Dense NVFP4 的额外收益；新增 `artifacts/results/dinov3_batch_speed_summary.csv`。
- 影响文件：`scripts/plot_end_to_end_speedup_summary.py`、`artifacts/results/end_to_end_speedup_summary.*`、`artifacts/results/dinov3_batch_speed_summary.csv`。
- 验证：`python3 -m py_compile` 通过；重新生成并预览 PNG，确认不再标注具体 img/s 数量，右侧柱子视觉问题已消除。

## 2026-05-18 - Refresh DINOv3 Sparse BF16 Sweep Figure
- 开发目的：纳入补测后的 DINOv3 4:8 Sparse BF16 batch sweep 速度结果，并简化右侧展示。
- 修改内容：右侧从两个 DINOv3 小图改为一个 `DINOv3 Speedup vs. Batch Size` 折线图；重新生成 `end_to_end_speedup_summary.*` 和 `dinov3_batch_speed_summary.csv`。
- 影响文件：`scripts/plot_end_to_end_speedup_summary.py`、`artifacts/results/end_to_end_speedup_summary.*`、`artifacts/results/dinov3_batch_speed_summary.csv`。
- 验证：`python3 -m py_compile` 通过；重新生成并预览 PNG。
- 后续注意：当前 Sparse BF16 sweep CSV 中已有 batch 1/2/4/8/32/64/128，尚未看到 batch 16 的 Sparse BF16 结果。
