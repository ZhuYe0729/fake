# Llama3.1 canonical prefill-only Pareto

Run the Llama3.1-8B-Instruct B=8/S=2048 prefill-only experiment in an isolated
058 bundle. Reuse the verified Llama3 canonical sparse states and prefill local
errors from 057, and reuse the Llama3 kernel-level speed support from 038.
First validate/recalibrate E2E speed; then collect pure-prefill phase-vLLM NLL
labels, refit the prefill-only quality proxy, solve, close the frontier, and
evaluate selected downstream-task points.
