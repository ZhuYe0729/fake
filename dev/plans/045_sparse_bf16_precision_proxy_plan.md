# Sparse BF16 Precision Proxy Modeling Plan

## Summary
- Create a new experiment directory: `artifacts/debug/016_llama2_sparse_bf16_precision_proxy`.
- Reuse source data from `014_llama2_prefill_loss_modeling`, but keep all new scripts, sampled policies, loss rows, fitted model outputs, and plots under `016`.
- Fit a sparse BF16 multiplicative sensitivity proxy:
  `pred_loss_delta = bias + sum(local_error_i * layer_coef[layer_i] * type_coef[type_i])`
- Produce a holdout trend plot showing predicted proxy vs measured downstream prefill loss delta.

## Key Changes
- Add experiment scripts under the new `016` directory:
  - policy sampling
  - multi-GPU sampled loss evaluation
  - proxy fitting and plotting
- Sampling budget: about 120 sparse BF16 configs.
  - compression counts: `4, 8, 16, 32, 64, 112, 168, 224`
  - `15` configs per count
  - mix uniform random samples and balanced layer/type samples
- Use one launcher command for parallel GPUs, defaulting to GPUs `1,2,3,4`.
  - The launcher starts one worker per GPU.
  - Workers evaluate disjoint policies.
  - The user does not manually run shard commands.

## Modeling
- Use sparse BF16 `output_rel_mse` from `014/sensitivity/module_method_local_errors.csv` as the default local error.
- Fit on train policies and validate on holdout policies:
  - deterministic `70% train / 30% holdout`
  - positive layer/type coefficients parameterized in log space
  - normalize type coefficients to geometric mean `1.0`
  - include a fitted bias
  - no quadratic terms or multi-linear nonlinear interaction terms
- Report Pearson, Spearman, MAE, and RMSE for train and holdout.
- Main figure uses holdout policies only and annotates holdout rank correlation.

## Test Plan
- Smoke run with a small policy count and one GPU to verify policy generation, replacement, loss evaluation, and CSV output.
- Multi-GPU dry run with GPUs `1,2,3,4` and limited policies to verify each policy is evaluated once.
- Full run with about 120 policies, then fit and generate the holdout plot.
- Validate that every selected module has sparse BF16 local error data and that dense baseline loss is consistent across workers.

## Assumptions
- Downstream loss means the existing WikiText-2 prefill mean CE loss setup.
- Conda env is `cospaq`; no SLURM/supercomputer handling is needed.
- Existing sparse BF16 prepared weights remain at `artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared/sparse_bf16/model.pt`.
- Existing dirty/untracked files are unrelated and must not be reverted.
- After implementation, append development notes to `dev/impls/045_sparse_bf16_precision_proxy_impl.md`.
