# Kernel NVFP4 Precision Proxy Extension Plan

## Summary
- Extend `artifacts/debug/016_llama2_sparse_bf16_precision_proxy` to also model `dense_nvfp4` and `sparse_nvfp4`.
- Use kernel-aware local errors from `015_llama2_prefill_kernel_loss_modeling`, where activation quantization is included by real runtime kernels.
- Reuse the existing 120 sampled policies from `016` for direct comparison across methods.
- Fit one multiplicative proxy per method:
  `pred_loss_delta = bias + sum(kernel_local_error * layer_coef[layer] * type_coef[linear_type])`.

## Key Changes
- Add a kernel-aware sampled loss runner under `016/scripts` that imports `015/scripts/common_kernel_prefill_loss.py`.
- Add a kernel-aware proxy fitter for `dense_nvfp4` and `sparse_nvfp4`.
- Keep outputs separated by method:
  - `loss/loss_samples_dense_nvfp4.csv`
  - `loss/loss_samples_sparse_nvfp4.csv`
  - `model/fitted_dense_nvfp4_proxy.json`
  - `model/fitted_sparse_nvfp4_proxy.json`
  - `plots/holdout_dense_nvfp4_proxy_vs_loss_delta.png`
  - `plots/holdout_sparse_nvfp4_proxy_vs_loss_delta.png`
- Update the `016` README with the NVFP4 workflow.

## Implementation Details
- Default local error metric is `output_rel_mse` from `015/sensitivity/module_method_kernel_local_errors.csv`.
- Loss evaluation installs real kernel modules with `install_kernel_modules()`:
  - `dense_nvfp4`: `NVFP4Linear`
  - `sparse_nvfp4`: `PaddedSparseNVFP4Linear`
- Multi-GPU launcher uses one worker per visible GPU and supports `--skip-existing`.
- Fitting uses the same deterministic 70/30 train/holdout split as sparse BF16.

## Test Plan
- Run syntax checks for new scripts.
- Run a small GPU smoke with `--max-policies 2` for both NVFP4 methods.
- For full evaluation, run both methods over all 120 sampled policies, then fit both proxies.
- Verify each method has 120 unique policies, generated metrics, and holdout plots.

## Assumptions
- Sparse BF16 results remain unchanged.
- NVFP4 sampled policies reuse `policies/sampled_sparse_bf16_policies.csv`.
- Full loss evaluation should use GPUs `1,2,3,4` via `CUDA_VISIBLE_DEVICES=1,2,3,4`.
