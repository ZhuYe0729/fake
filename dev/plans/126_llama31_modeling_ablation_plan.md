# Llama3.1 modeling ablation plan

Use the completed Llama3.1-8B-Instruct B=8/S=2048/O=64 prefill-decode
measurements to run a compact, offline paper ablation.  Do not rerun GPU
evaluation or modify the frozen 057/060 results.

1. Fit three quality proxies on the fixed 54/18 policy split: a global local-
   error sum, a method/depth/type structure-aware proxy, and the full
   phase-aware nonnegative factorized proxy.  Verify the full variant matches
   the frozen 057 model.
2. Score the fixed 7/5 speed-calibration split with method-aware Roofline,
   shape-corrected kernel aggregation, and monotone policy-level E2E
   calibration.  Verify the corrected and calibrated variants match 057.
3. Emit reproducible raw predictions, compact held-out metric tables, a
   two-panel paper figure, and a short Markdown/paper-text report under a new
   isolated 064 debug directory.
4. Run the complete offline pipeline twice and verify deterministic outputs.

