# FakeVLM Layer Error vs Loss Gap

This directory visualizes an existing FakeVLM result from `../024_fakevlm_prefill_global_pareto`.

## Main output

- `fakevlm_layer_profile_error_vs_nll_proxy.png`
- `fakevlm_layer_profile_error_vs_nll_proxy.pdf`
- `fakevlm_layer_error_loss_gap.png`
- `fakevlm_layer_error_loss_gap.pdf`

## What it shows

For `sparse_nvfp4`, `model.language_model.layers.18.mlp.up_proj` and `model.language_model.layers.22.self_attn.v_proj` have local `output_rel_mse` within 1.079x, but the fitted per-module NLL-delta proxy differs by 20.54x.

The right-hand loss-impact value is not a newly measured single-layer NLL. It is the per-module quality cost from the already fitted `024` quality model, which was trained against measured full-model NLL rows in `quality/stratified_loss.csv` using the valid `assistant_answer_token_nll_v2_active_prefix_aligned` definition.

## Method stats

- `dense_nvfp4`: layer mean local-error CV=0.169, layer mean NLL-proxy CV=0.165, max module NLL-proxy / median=1.72x.
- `sparse_bf16`: layer mean local-error CV=0.241, layer mean NLL-proxy CV=0.222, max module NLL-proxy / median=1.58x.
- `sparse_nvfp4`: layer mean local-error CV=0.217, layer mean NLL-proxy CV=0.572, max module NLL-proxy / median=18.32x.

## Supporting files

- `layer_method_summary.csv`: per-method, per-layer local error and fitted quality-cost summary.
- `close_error_examples.csv`: close-local-error module pairs with large quality-cost differences.
- `policy_loss_summary.csv`: measured full-model NLL deltas for stratified method-ratio policies.
- `policy_accuracy_summary.csv`: measured FakeClue accuracy deltas for the same stratified policies.
- `sparse_nvfp4_layer_profile_only.csv`: source summary for the standalone layer-profile plot.
- `summary.json`: compact machine-readable summary.
