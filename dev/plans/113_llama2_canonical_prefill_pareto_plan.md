# Llama2 canonical prefill Pareto

Use the completed 054 canonical-sparse quality model with the existing calibrated roofline latency predictor to solve prefill-only policies. Validate selected solved points using canonical phase export, real vLLM NLL and speed, then produce updated Pareto artifacts. Legacy direct-prune results are not reused as quality evidence.
