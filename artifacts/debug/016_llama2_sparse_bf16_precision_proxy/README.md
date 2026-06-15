# Llama2 Sparse BF16 Precision Proxy

This experiment fits a sparse BF16 precision proxy from sampled multi-linear
compression policies and downstream WikiText-2 prefill loss deltas.

It intentionally lives outside `014_llama2_prefill_loss_modeling`; `014` is used
only as the source of existing per-linear sparse BF16 local error data and common
prefill loss helpers.

## Workflow

Generate sampled policies:

```bash
python artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/generate_sparse_bf16_policies.py
```

Run sampled loss evaluation on GPUs 1,2,3,4 with one launcher command:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/run_sparse_bf16_loss_samples.py \
  --skip-existing
```

Fit the proxy and create the holdout trend plot:

```bash
MPLCONFIGDIR=/tmp/mplconfig python artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/fit_sparse_bf16_proxy.py
```

Run kernel-aware dense/sparse NVFP4 sampled loss evaluation with real runtime
kernels and activation quantization:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/run_kernel_nvfp4_loss_samples.py \
  --methods dense_nvfp4,sparse_nvfp4 \
  --skip-existing
```

Fit the kernel-aware NVFP4 proxies:

```bash
MPLCONFIGDIR=/tmp/mplconfig python artifacts/debug/016_llama2_sparse_bf16_precision_proxy/scripts/fit_kernel_nvfp4_proxy.py \
  --methods dense_nvfp4,sparse_nvfp4
```

## Outputs

- `policies/sampled_sparse_bf16_policies.csv`
- `loss/loss_samples_sparse_bf16.csv`
- `model/fitted_sparse_bf16_proxy.json`
- `model/predictions_sparse_bf16.csv`
- `model/proxy_metrics_sparse_bf16.csv`
- `plots/holdout_proxy_vs_loss_delta.png`
- `summary/README.md`
- `loss/loss_samples_dense_nvfp4.csv`
- `loss/loss_samples_sparse_nvfp4.csv`
- `model/fitted_dense_nvfp4_proxy.json`
- `model/fitted_sparse_nvfp4_proxy.json`
- `plots/holdout_dense_nvfp4_proxy_vs_loss_delta.png`
- `plots/holdout_sparse_nvfp4_proxy_vs_loss_delta.png`
