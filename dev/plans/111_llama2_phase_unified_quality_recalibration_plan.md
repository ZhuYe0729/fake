# Llama2 phase-unified quality recalibration plan

- Rebuild the Llama2 prefill-only 72-policy real-vLLM NLL dataset in an isolated 053 bundle.
- Use phase-degenerate exports for every compressed policy and retain BF16 as the raw dense reference.
- Keep the 046 feature model and frozen train/holdout split unchanged; report direct 046-versus-053 metrics.
- Do not solve a new Pareto frontier or update exported paper artifacts in this plan.
