# Llama2-7B vLLM Uniform Baseline

Speed scenario: `prefill_plus_1_decode` (`batch_size=16`, `prompt_len=1024`, `max_tokens=1`, eager vLLM, prefix cache disabled).

Quality source: 018 full ARC-Challenge uniform results (`arc_sample_len=1172`), not limit-128.

| method | median ms | speedup | tok/s | NLL delta | ARC-C acc_norm |
|---|---:|---:|---:|---:|---:|
| dense_bf16 | 1051.619 | 1.000 | 15579.8 | 0.0000 | 0.4514 |
| dense_nvfp4 | 563.644 | 1.866 | 29068.0 | 0.0820 | 0.4377 |
| sparse_bf16 | 631.798 | 1.664 | 25932.4 | 0.3503 | 0.3379 |
| sparse_nvfp4 | 504.653 | 2.084 | 32465.9 | 1.3184 | 0.2287 |

Notes:
- vLLM generate API was measured with one generated token, so this is a stable prefill-only approximation rather than strict `output_tokens=0`.
- Custom quant methods currently require `enforce_eager=True`; dense BF16 was also measured with eager mode for a consistent baseline.
