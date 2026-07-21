# Llama2 prefill-only vLLM runtime quality plan

Replace the invalid HFLM weight-proxy measurements with real vLLM likelihood
evaluation for all Llama2 prefill-only uniform and published ours policies.
Reuse existing speed results and existing phase-heterogeneous checkpoints.

1. Create debug experiment `042_llama2_prefill_only_vllm_runtime_quality` with a
   manifest that resolves every runtime checkpoint and checks each reused ours
   checkpoint policy equals the final published policy.
2. Use lm-eval's VLLM adapter with vLLM prompt logprobs for WikiText rolling
   PPL and 0-shot Winogrande, ARC-Easy, ARC-Challenge, and MMLU likelihood.
3. Audit actual runtime logprobs for five uniform methods plus `ours_point_012`.
4. Run all 14 policies resumably, then join their vLLM quality with existing
   measured prefill-only speeds in debug-only tables and plots.
5. Keep Llama3 and prefill-decode runtime-NLL work as TODO; do not mutate main
   exports or remeasure speed.
