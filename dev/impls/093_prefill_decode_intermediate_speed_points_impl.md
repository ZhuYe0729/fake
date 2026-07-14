## 2026-07-13 - Intermediate speed-gap refinement
- 开发目的：填补 prefill-decode 实测曲线 point 8（1.179x）至 max-speed point 11（1.714x）之间的速度空档。
- 修改内容：在独立 036 debug 根目录将 quality-budget grid 从原求解器的 12 个唯一点加密为 36 个候选，正式测试后发现 raw latency 对中段偏乐观；随后将 point 34→35 的 11 个离散 `o_proj` 切换拆分为 3/6/9 个 module 的点 36/37/38。
- 测量结果：point 34 的 10 样本速度为 1.4511x、真实 WikiText ΔNLL=2.0182。点 36/37/38 出现分钟级 phase-runtime stall；按用户决定排除 >10s 明确异常后，正常样本中位速度/ΔNLL 分别为 1.6267x/2.0497（6 样本）、1.6595x/2.0840（7 样本）、1.6803x/2.1094（6 样本）。四点均在实测 NLL—速度空间非支配。
- 影响文件：`artifacts/debug/036_llama2_prefill_decode_intermediate_points/` 下的 policies、checkpoint、formal speed、actual NLL、`report/intermediate_actual_nll_summary.csv` 和 `pareto_speedup_vs_wikitext_with_intermediates.png`。
- 后续注意：这些新点标记为 `stall-screened`，不应替代干净连续-runner/独立进程复测后的正式数值；当前只用于证明离散大跳变可被 module-level refinement 填补。
