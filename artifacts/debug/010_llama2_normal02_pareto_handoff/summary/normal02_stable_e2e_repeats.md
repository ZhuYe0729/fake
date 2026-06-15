# Llama2 Normal-02 Stable E2E Repeats

Scenario: batch_size=1, input_tokens=16384, output_tokens=256

Method: one fresh Python process per repeat, `iters_per_process=1`, `repeats=3`, GPU7.

| Point | OK/Attempts | Pred Total (ms) | E2E Mean (ms) | Std (ms) | Speedup vs Point0 | Backends |
|-------|-------------|-----------------|---------------|----------|-------------------|----------|
| 0 | 3/3 | 4176.5 | 9026.0 | 4.5 | 1.0000 | {'bf16': 224} |
| 7 | 3/3 | 3190.3 | 8340.8 | 12.4 | 1.0821 | {'bf16': 153, 'dense_nvfp4/marlin_nvfp4': 71} |
| 9 | 3/3 | 2829.1 | 7394.2 | 5.5 | 1.2207 | {'marlin_nvfp4': 128, 'dense_nvfp4/marlin_nvfp4': 96} |

## Interpretation

- Stable repeat ranking matches predicted ranking: point_9 < point_7 < point_0 latency.
- point_9 remains the best validated operating point: about 1.22x E2E speedup vs dense with NLL delta +0.0368 from the quality validation.
- Process-per-repeat avoids the OOM seen with long single-process repeat runs.
