# Prefill-Only Recommendation

## Conclusion

Strictly requiring all three conditions below does not currently yield a strong Llama2-7B vLLM prefill-only showcase:

- Heterogeneous compression is clearly faster than every uniform compressed method.
- Accuracy loss is not large.
- The workload is a meaningful prefill-only setting rather than a tiny overhead-dominated case.

The best already measured prefill-only proxy is still `batch=8, input_seq=512, output_seq=1` with `max_speed_hetero`:

| scenario | policy | vs best uniform | best uniform | acc_norm | dense acc_norm | acc_norm loss |
|---|---|---:|---|---:|---:|---:|
| `b8_in512_out1` | `maxspeed_004_f2600ffcfc` | `1.020x` | `sparse_nvfp4` | `0.4087` | `0.4514` | `0.0427` |

This is the safest prefill-only candidate with real vLLM speed and full ARC-Challenge quality already measured, but the speed advantage is not visually strong.

## Why Larger Pure-Prefill Cases Are Weak

For practical filters like `batch>=8` or `prefill_m=batch*input_seq>=4096`, the P024 quality-constrained hetero policy is predicted to be slower than the best uniform method:

| practical pattern | best uniform | P024 optimized vs best uniform | max-speed vs best uniform | max-speed quality cost |
|---|---|---:|---:|---:|
| `prefill_m=16384` examples | `sparse_nvfp4` | `0.872x` | `1.155x` | `1.054` |
| `prefill_m=8192` examples | `sparse_nvfp4` | `0.842x` | `1.107x` | `1.054` |

The speed-only max-speed policy mixes `sparse_bf16` and `sparse_nvfp4`, so it does show method heterogeneity, but its quality cost is far above the P024 budget `0.296`. Existing max-speed quality results with lower quality cost already drop to `acc_norm=0.3362` or `0.2884`, so this is not a good accuracy-preserving showcase.

## If a Prefill-Only Candidate Must Be Tested

The only pure-prefill prediction where P024 optimized hetero clearly beats every uniform method is tiny:

| scenario | best uniform | predicted P024 optimized vs best uniform | predicted max-speed vs best uniform | P024 method mix |
|---|---|---:|---:|---|
| `b1_in128_out1` | `marlin_nvfp4` | `1.186x` | `1.242x` | `dense_bf16:73,sparse_bf16:55` |

This should be treated as a hypothesis, not a final claim, because `batch=1,input_seq=128` is likely vLLM overhead dominated. The generated focused retest script includes this case.

## Recommendation

- For a paper/demo figure that must be prefill-only, use `batch=8,input_seq=512,output_seq=1,max_speed_hetero` only if a modest `1.020x` over best uniform is acceptable.
- For a stronger method story, prefer the already selected prefill-decoding scenarios `batch=8,input_seq=16384,output_seq=64/128`, where optimized hetero is `2.777x/3.550x` faster than the best uniform method.
- Do not claim a strong prefill-only advantage for meaningful large-prefill workloads without new evidence; current modeling says the quality-constrained hetero policy loses to uniform `sparse_nvfp4`.
