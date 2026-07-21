# q120 critical-module ablation and phase-interface control

All values are direct vLLM prompt-logprob NLL over the fixed 100×2048 WikiText protocol.

| configuration | runtime quantization path | ΔNLL vs BF16 |
|---|---|---:|
| dense BF16 | none | 0 |
| `q120` | `phase_hetero_mytest`, 120 prefill NVFP4 | 0.000304 |
| eight q120 leave-one policies | `phase_hetero_mytest`, 121 prefill NVFP4 | 0.000269–0.000350 |
| `q128_phase` | `phase_hetero_mytest`, 128 prefill NVFP4 | 0.000392 |
| `p01` uniform dense-NVFP4 | `nvfp4_mytest` | 0.053822 |

`q128_phase - q120 = 0.0000876`: the eight modules are neither individually nor jointly responsible for the 0.0538 NLL gap.  The q128 phase trace records `apply_prefill: 128`, so the phase dispatcher did select the 128 prefill methods.

## Conclusion

The uniform and phase-heterogeneous paths do not presently have equivalent NVFP4 NLL semantics.  Consequently, the prior comparison between uniform `p01` and mixed phase policies cannot diagnose the quality model or establish a Pareto advantage.  Before changing the precision proxy again, verify the phase-heterogeneous dense-NVFP4 operator/export path against the uniform `nvfp4_mytest` operator on an all-NVFP4 policy.
