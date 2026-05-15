## 2026-05-15 - FlashInfer Custom Shape Benchmark
- 开发目的：新增纯 `(m,n,k)` FlashInfer NVFP4 microbenchmark，测试 activation quant、FP4 GEMM、forward-like 与 dense Linear bf16/fp32 baseline 的性能。
- 修改内容：新增 custom shape benchmark 脚本、结果分析脚本、Slurm analysis 入口，并补充脚本 README。
- 影响文件：
  - `dev/plans/010_flashinfer_custom_shape_plan.md`
  - `scripts/bench_flashinfer_custom_shapes.py`
  - `scripts/analyze_flashinfer_custom_shapes.py`
  - `scripts/slurm/analysis/bench_flashinfer_custom_shapes.sh`
  - `scripts/README.md`
  - `dev/impls/010_flashinfer_custom_shape_impl.md`
- 后续注意：正式 benchmark 需要提交到 GPU 节点；`large` preset 可能包含超大 shape，OOM 时会写入 `status=ERROR`。

## 2026-05-15 - Balanced Run Status Analysis
- 开发目的：核查 `PRESET=balanced WARMUP=20 ITERS=100 sbatch scripts/slurm/analysis/bench_flashinfer_custom_shapes.sh` 是否完整完成，并解释末尾 `OSError`。
- 修改内容：检查 `out/flashinfer_shapes_442104.out`、`err/flashinfer_shapes_442104.err`、`artifacts/analysis/flashinfer/custom_shapes.csv`；确认 67 个 planned shapes 中 47 个完整完成，第 48 个 shape 部分写入后因 `[Errno 122] Disk quota exceeded` 中断，后续 analysis 未运行。
- 影响文件：
  - `out/flashinfer_shapes_442104.out`
  - `err/flashinfer_shapes_442104.err`
  - `artifacts/analysis/flashinfer/custom_shapes.csv`
  - `dev/impls/010_flashinfer_custom_shape_impl.md`
- 后续注意：释放 quota 或改写 `OUTPUT` 到有配额的位置后，可从 `compute_reuse_sweep m=4096 n=1024 k=4096` 起补跑剩余 shapes；当前 `custom_shapes.csv` 只适合分析前 47 个完整 shape。

## 2026-05-15 - Balanced Result Merge and Analysis
- 开发目的：合并失败前备份结果与从失败 shape 续跑的结果，补齐完整 balanced benchmark 的整理分析。
- 修改内容：将 `custom_shapes_bk.csv` 的 shape index `0-46` OK rows 与 `custom_shapes_resume_from_47.csv` 的续跑 OK rows 合并；续跑 rows 按顺序重映射为原 balanced shape index `47-66` 并恢复 shape family；重新生成 summary、family/component CSV、三张图，并新增人工解读报告。
- 影响文件：
  - `artifacts/analysis/flashinfer/custom_shapes.csv`
  - `artifacts/analysis/flashinfer/custom_shapes_merged.csv`
  - `artifacts/analysis/flashinfer/summary.csv`
  - `artifacts/analysis/flashinfer/speedup_by_shape_family.csv`
  - `artifacts/analysis/flashinfer/component_breakdown.csv`
  - `artifacts/analysis/flashinfer/summary.md`
  - `artifacts/analysis/flashinfer/speedup_vs_m.png`
  - `artifacts/analysis/flashinfer/speedup_vs_arithmetic_intensity.png`
  - `artifacts/analysis/flashinfer/quant_gemm_breakdown.png`
  - `artifacts/analysis/flashinfer/custom_shapes_analysis.md`
  - `dev/impls/010_flashinfer_custom_shape_impl.md`
- 后续注意：合并后完整覆盖 67/67 shapes、603 条 OK op rows、0 ERROR rows；续跑文件本身的 `shape_family=custom` 不应直接用于 family-level 分析，应使用合并后的 `custom_shapes.csv`。

## 2026-05-15 - Top-Level Summary Link
- 开发目的：在总分析报告中补充 FlashInfer 细粒度 custom shape 分析入口。
- 修改内容：更新 `artifacts/analysis/summary.md`，说明 FlashInfer NVFP4 纯 `(m,n,k)` 细粒度分析位于 `artifacts/analysis/flashinfer/`，入口报告为 `custom_shapes_analysis.md`。
- 影响文件：
  - `artifacts/analysis/summary.md`
  - `dev/impls/010_flashinfer_custom_shape_impl.md`
- 后续注意：总报告只保留入口说明，详细数据和图表继续维护在 `flashinfer/` 子目录。

## 2026-05-15 - Fixed-Dimension Sweep Visualization
- 开发目的：补充 FlashInfer balanced preset 中固定两个维度、遍历第三个维度的 speedup 可视化。
- 修改内容：在 `scripts/analyze_flashinfer_custom_shapes.py` 中新增 `fixed_dimension_sweep_speedups.png`，包含 `m/n/k` large/small 共 6 个子图，每个子图同时展示 NVFP4 forward-like 相对 dense bf16 与 dense fp32 的加速曲线；重新生成 FlashInfer 分析输出，并在人工分析报告中加入图和说明。
- 影响文件：
  - `scripts/analyze_flashinfer_custom_shapes.py`
  - `artifacts/analysis/flashinfer/fixed_dimension_sweep_speedups.png`
  - `artifacts/analysis/flashinfer/summary.md`
  - `artifacts/analysis/flashinfer/custom_shapes_analysis.md`
  - `dev/impls/010_flashinfer_custom_shape_impl.md`
- 后续注意：`k_sweep_large_context` 的含义是固定 `m=4096,n=4096` 并遍历 `k`；small context 对应固定 `m=512,n=512` 并遍历 `k`。

## 2026-05-15 - Fixed-Dimension Breakdown Visualization
- 开发目的：补充与 fixed-dimension speedup 图对应的详细 latency breakdown 可视化。
- 修改内容：在 `scripts/analyze_flashinfer_custom_shapes.py` 中新增 `fixed_dimension_sweep_breakdown.png`，同样包含 6 个 fixed-dimension 子图；每个 shape 使用 stacked bar 展示 activation scale+quant、NVFP4 GEMM-only 和剩余 forward overhead，并更新 FlashInfer summary 与人工分析报告。
- 影响文件：
  - `scripts/analyze_flashinfer_custom_shapes.py`
  - `artifacts/analysis/flashinfer/fixed_dimension_sweep_breakdown.png`
  - `artifacts/analysis/flashinfer/summary.md`
  - `artifacts/analysis/flashinfer/custom_shapes_analysis.md`
  - `dev/impls/010_flashinfer_custom_shape_impl.md`
- 后续注意：`other forward overhead` 按 `nvfp4_forward_like_ms - activation_scale_plus_quant_ms - nvfp4_gemm_only_ms` 计算，并截断为非负值，表示未单独归因的剩余耗时。
