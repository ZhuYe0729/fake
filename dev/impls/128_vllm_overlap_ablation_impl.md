# vLLM phase-switch overlap ablation implementation log

## 2026-07-22 - Llama3.1 runtime overlap ablation

- Fixed the experiment to Llama3.1-8B-Instruct `point_011` under the main
  B=8/S=2048/O=64 phase-vLLM protocol; all 128 Linear modules switch methods.
- Added policy/config audit, canonical checkpoint materialization, paired AB/BA
  latency runner, exclusive-run locking, GPU-idle checks, resumable raw CSV,
  plotting, Markdown report, JSON summary, and LaTeX table.
- Retained seven formal sample pairs with tracing disabled. Both variants in
  every pair ran on the same GPU and every retained run began at 4 MiB GPU
  usage and 0% utilization.
- Result: synchronous boundary copy 1625.87 ms median versus asynchronous
  overlap 1521.73 ms median, a 6.41% E2E latency reduction (1.068x). Overlap
  wins 5/7 paired comparisons; one async long-tail sample is retained.
- During collection, an external process caused one OOM and an earlier runner
  remained active after a tool-session yield. Those incomplete/concurrent
  samples were rejected, the runner was given an exclusive lock, and pairs
  3--7 were recollected serially before final validation.
- Main artifacts: `artifacts/debug/066_llama31_vllm_overlap_ablation/`.

## 2026-07-22 - Direct overlap-effectiveness profile

- Added a diagnostic-only monkey patch that wraps the existing decode-state
  materialization and first-use wait functions with CUDA events; the vLLM
  source and formal runtime path remain unchanged.
- Measured three independent executions of the same Llama3.1 `point_011`
  workload. The median total transfer work is 354.94 ms, exposed copy stall is
  1.92 ms, and hidden transfer work is 353.02 ms.
- The median per-run hidden ratio is 99.38% (reported as 99.4% in prose),
  directly validating that nearly all phase-switch transfer work overlaps with
  useful computation.
- Added `report/overlap_profile.{md,json}` and
  `report/overlap_profile_table.tex`, and integrated the mechanism numbers into
  the main summary and paper-facing paragraph.
