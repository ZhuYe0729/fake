# Llama2 Full-Model Trace Oracle

## Purpose

This debug run tries to derive an oracle policy for `llama2-7b normal_02` from full-model execution, not standalone module benchmarks.

It traces each candidate method inside the real full model, aggregates per-linear-group prefill and decode latency, builds a trace oracle policy, and validates that policy with no-hook E2E.

## Files

- `scripts/trace_full_model_method.py`: traces one full-model method.
- `scripts/build_trace_oracle_policy.py`: builds a policy from traced group rankings.
- `scripts/eval_oracle_e2e.py`: evaluates a policy with normal no-hook E2E.
- `scripts/generate_attention_ablation_policies.py`: builds the final k/q/v ablation policies.
- `results/{method}/group_projection.csv`: traced group latency projection for each method.
- `results/oracle/oracle_policy.csv`: initial module-trace oracle policy.
- `results/oracle/oracle_full_e2e.csv`: no-hook E2E for the initial trace oracle.
- `results/attention_ablation_summary.csv`: direct full-model E2E ablation over k/q/v choices.
- `results/refined_oracle/refined_oracle_policy.csv`: final refined oracle policy.
- `results/e2e_comparison.csv`: comparison against previous manual/pred/single results.

## Result

The initial module-trace oracle selected:

- MLP: `dense_nvfp4->marlin_nvfp4`
- `self_attn.o_proj`: `marlin_nvfp4`
- `self_attn.k_proj/q_proj/v_proj`: `dense_bf16`

Its no-hook E2E was `7729.26 ms`, worse than pred. This shows that hook-based per-module timing still misranks k/q/v for full E2E.

The refinement ablation fixed MLP as hybrid and `o_proj` as Marlin, then directly tested all k/q/v bf16-vs-Marlin combinations. The best ablation was:

- MLP: `dense_nvfp4->marlin_nvfp4`
- attention q/k/v/o: `marlin_nvfp4`

This refined oracle policy is identical to the existing pred policy.

## E2E Summary

| source | e2e ms |
|---|---:|
| previous pred | 7282.37 |
| refined oracle / ablation best | 7426.74 |
| single marlin | 7717.87 |
| manual | 7722.88 |
| initial trace oracle | 7729.26 |

The refined oracle and pred policies are identical; the E2E difference comes from different run instances. Within the ablation sweep, the all-Marlin attention variant was the best.

## Conclusion

For `llama2-7b normal_02`, the best policy found from full-model evidence is the same as pred:

- `mlp.down_proj`: `dense_nvfp4->marlin_nvfp4`
- `mlp.gate_proj`: `dense_nvfp4->marlin_nvfp4`
- `mlp.up_proj`: `dense_nvfp4->marlin_nvfp4`
- `self_attn.q_proj/k_proj/v_proj/o_proj`: `marlin_nvfp4`

The full-model trace is useful for narrowing candidates, but final oracle selection still needs direct no-hook full-model ablation for groups where hook trace and E2E disagree.
