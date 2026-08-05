# Llama3.1 quality calibration-size sensitivity plan

Use the frozen 057 Llama3.1 prefill-decode quality labels to quantify how the
full phase-aware quality surrogate changes with the number of calibration
policies. Do not rerun GPU evaluation or alter the fixed 18-policy holdout.

1. Build nested calibration subsets of sizes 5/8/12/18/27/36/45/54. Keep the
   five dense/uniform anchors in every subset; use deterministic farthest-point
   feature coverage for the main curve and ten fixed random nested orders for
   an IQR band.
2. Refit the unchanged full phase-aware ReLU model at every size and report
   holdout MAE, RMSE, and Spearman. Verify the 54-policy endpoint reproduces
   the frozen 064 full-model metrics.
3. Use each policy's recorded real-vLLM NLL elapsed time to report cumulative
   calibration GPU-minutes, and record CPU fitting time separately.
4. Predefine the elbow as the smallest size whose median holdout MAE is within
   5% of the 54-policy result and remains within that range thereafter. Report
   the observed elbow without retuning the rule.
5. Produce raw/summary CSVs, a paper-facing sensitivity curve, Markdown report,
   and draft-ready text in a new isolated 065 debug directory.
