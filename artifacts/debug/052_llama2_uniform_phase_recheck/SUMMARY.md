# Uniform NVFP4 versus phase-homogeneous NVFP4 forensic check

## Controlled setup

All values below are fixed WikiText prompt-logprob NLL over the first 20
2048-token blocks. The comparison holds the source model and NVFP4 packed
format fixed where stated.

| checkpoint / runtime | source weights before packing | avg NLL | delta versus BF16 |
| --- | --- | ---: | ---: |
| BF16 (`p00`) | original BF16 | 1.999849 | 0.000000 |
| phase q128 | original BF16 | 2.001808 | 0.001959 |
| uniform direct control | original BF16 | 2.042132 | 0.042283 |
| historical uniform `p01` | prepared `dense_nvfp4` | 2.049706 | 0.049857 |

`uniform direct control` is exported by
`scripts/export_uniform_from_original.py`. Its checkpoint uses the regular
`nvfp4_mytest` runtime, but bypasses the historical prepared state.

## Findings

1. The historical uniform dense-NVFP4 baseline is not a single direct packing
   of the original BF16 weights. `prepare_uniform_compressed.py` first runs
   `nvfp4_quantize_weight(...)` with activation/Hessian calibration and writes
   its fake-quantized/dequantized result into `prepared/dense_nvfp4/model.pt`.
   `export_uniform_vllm.py` then invokes `quantize_weight(...)` again when it
   packs that prepared state for vLLM. The prepared state differs from the
   original BF16 weights; e.g., L0 `mlp.down_proj` has RMSE `0.001471` before
   the second pack.
2. Repacking this prepared state accounts for `0.007574` average NLL on this
   control, or 15.2% of the historical uniform loss versus BF16.
3. The larger difference is runtime semantics. A uniform checkpoint exported
   directly from original weights and a phase q128 checkpoint have exactly the
   same 128 packed NVFP4 weight tensors and exactly the same 256 scale tensors,
   byte-for-byte. Yet their NLL differs by `0.040324`. Thus the remaining
   80.9% of the historical-uniform loss comes from the distinction between
   `nvfp4_mytest` and `phase_hetero_mytest` execution (including their runtime
   activation/linear-kernel path), not policy selection or static weight
   quantization.

## Consequence

The current paper-style prefill-only NLL / downstream-quality comparison is
not apples-to-apples if it compares historical uniform `nvfp4_mytest` with
ours under `phase_hetero_mytest`: a fully uniform policy evaluated through the
phase runtime is substantially more accurate than the historical uniform
baseline. Speed results remain measurements of their respective deployed
backends, but a quality Pareto plot must explicitly choose one of these two
semantics before making a method claim.

## Runtime-speed control

The speed control uses one warmup and five fresh-process runs of the paper
prefill-only workload (B=8, S=2048, O=1). Both checkpoints are directly packed
from the original BF16 model and use all 128 modules as dense NVFP4 in both
prefill and decode.

| runtime | five-run median E2E ms | mean ± std ms | relative to uniform |
| --- | ---: | ---: | ---: |
| `nvfp4_mytest` uniform | 662.638 | 663.203 ± 7.077 | 1.000x |
| `phase_hetero_mytest` degenerate-uniform | 656.298 | 660.642 ± 11.701 | 1.010x faster |

Therefore the phase dispatcher itself has no material prefill-only throughput
penalty in this all-NVFP4 control (the median difference is 0.96%, below the
run-to-run spread). An earlier control that used NVFP4 only for prefill but
kept decode BF16 measured 1068.951 ms and OOMed at 0.9 GPU-memory utilization:
that is a *different representation* which stores two full weight copies, not
dispatcher overhead. It must not be used as the uniform baseline.

The same direct control was run for the prefill-decode paper workload
(B=16, S=2048, O=80):

| runtime | five-run median E2E ms | mean ± std ms | relative to uniform |
| --- | ---: | ---: | ---: |
| `nvfp4_mytest` uniform | 4490.281 | 4458.699 ± 58.458 | 1.000x |
| `phase_hetero_mytest` degenerate-uniform | 4400.770 | 4401.758 ± 33.701 | 1.020x faster |

Thus phase selection also has no measurable E2E penalty when both phases use
the same method. The small 2.0% median difference is within the cross-GPU
repeat variation and is not claimed as a speedup.

## Remaining narrow check

To name the exact phase-runtime operation responsible for the `0.040324` NLL
gap, inspect the external vLLM implementations of `nvfp4_mytest` and
`phase_hetero_mytest`, specifically their activation quantization / dispatch
path. This is a backend-equivalence issue, not a precision-model fitting
issue.
