# Llama2 canonical sparse quality recalibration

Rebuild the Llama2 prefill-only quality labels using canonical SparseGPT-calibrated sparse weights.  Sparse BF16 is packed without re-pruning; sparse NVFP4 is calibrated as sparse BF16 and quantized exactly once by the phase exporter.  Preserve the fixed 72-policy design and the original local+global positive quality proxy.  Do not run Pareto solving, downstream tasks, or speed benchmarks in this plan.
