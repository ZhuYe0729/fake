# Pareto TODO

Pareto policy search is intentionally not implemented in the max-speed workflow.
The future input is a scenario-specific per-fused-Linear quality-cost table keyed
by the `phase_hetero_policy.json` module names. The solver will minimize the
same predictor latency objective subject to a summed quality-cost budget and
emit the same phase-heterogeneous policy format.
