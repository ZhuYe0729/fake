# 010 FlashInfer Custom Shape Benchmark Plan

## Summary

新增纯 `(m,n,k)` shape sweep，用于测试 FlashInfer NVFP4 activation quant、FP4 GEMM、forward-like 总耗时，并对比 dense `bf16` 与 dense `fp32` Linear baseline。结果输出到 `artifacts/analysis/flashinfer/`。

## Key Changes

1. 新增 `scripts/bench_flashinfer_custom_shapes.py`。
   - 支持 `--preset smoke|balanced|large` 和 `--shapes MxNxK ...`。
   - 默认输出 `artifacts/analysis/flashinfer/custom_shapes.csv`。
   - 对每个 shape 构造 activation `x=(m,k)` 和 dense weight `w=(n,k)`。
   - 每个 shape 记录 `dense_linear_bf16`、`dense_linear_fp32`、`activation_global_scale`、`activation_quant_only`、`activation_scale_plus_quant`、`weight_scale_plus_quant_once`、`alpha`、`nvfp4_gemm_only`、`nvfp4_forward_like`。
   - dense baseline 统一使用 `torch.nn.functional.linear(x, w)`；不使用裸 `a @ b.T`。
2. `balanced` preset 覆盖：
   - `m_sweep_large_context` / `m_sweep_small_context`：固定另两个维度，使用相同的 M 列表。
   - `n_sweep_large_context` / `n_sweep_small_context`：固定另两个维度，使用相同的 N 列表。
   - `k_sweep_large_context` / `k_sweep_small_context`：固定另两个维度，使用相同的 K 列表。
   - `compute_reuse_sweep`、`square_scale_sweep` 和若干现有模型代表 shape。
3. 新增 `scripts/analyze_flashinfer_custom_shapes.py`。
   - 生成 `summary.csv`、`speedup_by_shape_family.csv`、`component_breakdown.csv`。
   - 生成 `speedup_vs_m.png`、`speedup_vs_arithmetic_intensity.png`、`quant_gemm_breakdown.png`。
   - 输出 `artifacts/analysis/flashinfer/summary.md`。
4. 新增 `scripts/slurm/analysis/bench_flashinfer_custom_shapes.sh`。
   - 使用 `wja-cospaq`、`gpu_5090`，默认 benchmark 后运行 analysis。
   - 支持 `PRESET`、`SHAPES`、`WARMUP`、`ITERS`、`GEMM_BACKEND`、`QUANT_BACKEND`、`SF_LAYOUT`、`OUTPUT`。
5. 更新 `scripts/README.md`，加入用途、输出路径和冒烟命令。

## Test Plan

- 登录节点语法检查：
  - `python -m py_compile scripts/bench_flashinfer_custom_shapes.py scripts/analyze_flashinfer_custom_shapes.py`
- GPU 冒烟：
  - `PRESET=smoke WARMUP=2 ITERS=5 sbatch scripts/slurm/analysis/bench_flashinfer_custom_shapes.sh`
- GPU 正式均衡版：
  - `PRESET=balanced WARMUP=20 ITERS=100 sbatch scripts/slurm/analysis/bench_flashinfer_custom_shapes.sh`
- 验证 CSV、summary 和三张图都生成；非法或 OOM shape 写 `status=ERROR` 后继续。

## Assumptions

- small/large context 对应 sweep 使用完全相同的变化维度取值，便于后续逐点对比。
- NVFP4 主口径使用 bf16 activation/output；fp32 只作为 dense Linear baseline。
- dense baseline 使用 `F.linear(x, w)`，分别测试 bf16 和 fp32，不加 bias。
- 权重量化视为离线一次性成本，单独记录，不计入默认 `nvfp4_forward_like` speedup。
