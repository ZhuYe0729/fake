## 2026-05-14 - Analysis Summary
- 开发目的：汇总 `artifacts/analysis` 下 NVFP4 microbench 结果，按每个模型 unique shape/config 去重分析加速与减速情况。
- 修改内容：生成 unique shape 汇总 CSV、模型级 summary CSV、三张可视化图，并写入 `artifacts/analysis/summary.md`。
- 影响文件：
  - `artifacts/analysis/summary.md`
  - `artifacts/analysis/unique_shape_summary.csv`
  - `artifacts/analysis/model_speedup_summary.csv`
  - `artifacts/analysis/speedup_by_m.png`
  - `artifacts/analysis/speedup_vs_quant_share.png`
  - `artifacts/analysis/median_speedup_by_config.png`
- 后续注意：DINOv3 使用完整的 `microbench_bk.csv` 做统计；当前 `microbench.csv` 在分析时较小。若后续完成新的 DINOv3 结果，应重新生成 summary。

## 2026-05-14 - Quant/GEMM Breakdown
- 开发目的：进一步拆解 activation quant 与 FP4 GEMM 在不同模型、batch、resolution、M 桶下的耗时占比。
- 修改内容：新增 breakdown CSV 和图表，并在 `summary.md` 增加 `Activation Quant / GEMM Breakdown` 章节。
- 影响文件：
  - `artifacts/analysis/summary.md`
  - `artifacts/analysis/breakdown_by_model.csv`
  - `artifacts/analysis/breakdown_by_config.csv`
  - `artifacts/analysis/breakdown_by_m_bucket.csv`
  - `artifacts/analysis/breakdown_top_speedups.csv`
  - `artifacts/analysis/breakdown_worst_slowdowns.csv`
  - `artifacts/analysis/breakdown_quant_dominant.csv`
  - `artifacts/analysis/breakdown_gemm_dominant.csv`
  - `artifacts/analysis/breakdown_components_by_model.png`
  - `artifacts/analysis/quant_vs_gemm.png`
  - `artifacts/analysis/dinov3_quant_gemm_by_m.png`
- 后续注意：`activation_scale_plus_quant` 包含 global scale 和 activation quantize；该 breakdown 是 isolated layer microbench，不等同于端到端模型耗时拆分。

## 2026-05-14 - NVFP4 Layout Review
- 开发目的：检查当前 FlashInfer NVFP4 Linear 实现中 activation quant 和 GEMM 偏慢的可能原因。
- 修改内容：修正权重 FP4 tensor/scale tensor 转置后被 `.contiguous()` 变回 row-major 的问题；FlashInfer `mm_fp4` 要求 B 为 column-major，直接保留转置 view 的 stride。
- 影响文件：
  - `fake/kernels/flashinfer_nvfp4.py`
  - `dev/impls/009_analysis_summary_impl.md`
- 后续注意：已做 Python 语法检查；该改动需要在 GPU 节点上重新跑 `check_flashinfer_nvfp4.py` 或 microbench，确认 GEMM latency 是否下降。

## 2026-05-15 - MaxViT Large Summary Refresh
- 开发目的：根据 `artifacts/analysis/maxvit_large` 的新测试文件复核 summary 中 MaxViT large 的结果说明。
- 修改内容：确认 `microbench.csv` 与 `microbench_256_512_768.csv` 都没有 OK 性能样本，并在 `summary.md` 明确区分修正后的 224 倍数输入失败与旧 shape 口径被拒绝的结果。
- 影响文件：
  - `artifacts/analysis/summary.md`
  - `dev/impls/009_analysis_summary_impl.md`
- 后续注意：MaxViT large 目前仍不能纳入加速比统计；需要先解决 large 变体的 window partition 输入约束或专门选择可通过的输入尺寸。
