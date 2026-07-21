# Llama2 prefill mechanism-quality debug plan

## Goal

Improve the real-vLLM prefill NLL proxy without changing the main `046` result bundle.  The debug model separates quantization, sparsity, sparse accumulation, and sparse--quantization interaction, then re-solves a diagnostic prefill frontier.

## Decisions

- Work under `artifacts/debug/047_llama2_prefill_mechanism_quality_debug/`.
- Add 18 fixed real-vLLM NLL calibration policies: six quant-only, six sparse-only, three co-located sparse-NVFP4, and three separated dense-NVFP4 plus sparse-BF16 policies.
- Train with the existing 54 `046` train labels plus 12 mechanism policies; retain the old 18 holdout policies and six mechanism policies for validation.
- Use non-negative per-bucket/type quant and sparse factors, non-negative sparse-squared accumulation, and non-negative sparse--quantization interaction.  Keep the dense-BF16 prediction fixed at zero.
- Re-solve only after reporting model validation.  Treat uniform dense-NVFP4 as an explicit feasible anchor and emit a dense-only restricted frontier as a diagnostic.

## Success criteria

- Every new policy and NLL label is hash-checked and uses the identical 100 fixed WikiText blocks from `046`.
- The report separates old holdout, new mechanism holdout, and existing solved-policy challenge error.
- The new NLL plot includes uniform references and makes any dense-NVFP4 domination visible.
