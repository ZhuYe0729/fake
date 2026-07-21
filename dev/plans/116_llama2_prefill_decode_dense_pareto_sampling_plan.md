# 116 Llama2 prefill-decode dense Pareto sampling plan

## Goal

For the canonical Llama2-7B prefill-decode experiment, fill the missing measured-speed range between roughly 1.3x and 1.8x so the final Pareto figure shows a continuous quality/speed trade-off rather than disconnected clusters.

## Assumptions

- The established canonical quality proxy, policy action space, and roofline speed model remain unchanged.
- Formal speed uses the existing fresh-process runner with `gpu_memory_utilization=0.80` and robust median of repeated runs.
- Existing points and results are immutable references; new points are written under a separate `pareto/dense_speed/` namespace until closure succeeds.

## Plan

1. Audit the predicted action-space frontier and derive target speed bins from 1.30x to 1.80x.  
   Verify: each selected bin has a distinct policy signature and feasible action support.
2. Add a target-speed constrained DP solver that minimizes predicted quality loss subject to a minimum raw roofline speedup, rather than relying only on coarse quality budgets.  
   Verify: candidates cover the requested interval in prediction and are non-dominated under proxy quality/roofline speed.
3. Export and formally benchmark the selected candidates under the existing 0.80 fresh-process protocol.  
   Verify: retain only stable speed samples; discard OOM/interference runs.
4. Measure canonical real-vLLM NLL for retained candidates and select a non-dominated measured NLL/speed subset.  
   Verify: every plotted new point has both measured speed and measured NLL.
5. Run the three downstream generation tasks only for the retained subset, then merge into the existing task reports and redraw the figures.  
   Verify: full `1000/1500/333` coverage per policy/dataset and no mixed speed configuration in final plots.

## Deferred

- Re-solving the quality proxy or speed model.
- Evaluating point 007, which remains excluded due to unsupported/OOM speed closure.
