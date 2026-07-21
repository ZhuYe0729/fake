# Llama3 prefill speed decomposition

## Per-policy decomposition

| policy | predicted local (ms) | exact local (ms) | local error | E2E (ms) | E2E - exact local | E2E/exact |
|---|---:|---:|---:|---:|---:|---:|
| p00 | 1091.38 | 968.75 | 12.7% | 1294.82 | 326.07 | 1.337 |
| p01 | 527.73 | 501.88 | 5.2% | 700.13 | 198.26 | 1.395 |
| p02 | 529.67 | 578.77 | -8.5% | 1026.50 | 447.73 | 1.774 |
| p03 | 453.71 | 446.33 | 1.7% | 644.43 | 198.10 | 1.444 |
| p04 | 1043.42 | 990.64 | 5.3% | 1184.64 | 194.01 | 1.196 |
| point_000 | 1091.38 | 968.75 | 12.7% | 1294.20 | 325.45 | 1.336 |
| point_003 | 963.32 | 858.74 | 12.2% | 1184.74 | 326.00 | 1.380 |
| point_005 | 707.21 | 638.70 | 10.7% | 947.78 | 309.07 | 1.484 |
| point_007 | 602.36 | 566.24 | 6.4% | 825.91 | 259.67 | 1.459 |
| point_008 | 538.87 | 514.54 | 4.7% | 792.90 | 278.36 | 1.541 |
| point_009 | 484.00 | 452.99 | 6.8% | 788.60 | 335.62 | 1.741 |
| point_010 | 451.76 | 417.04 | 8.3% | 783.99 | 366.95 | 1.880 |
| point_011 | 427.05 | 395.43 | 8.0% | 768.46 | 373.03 | 1.943 |
| point_012 | 405.92 | 379.97 | 6.8% | 753.55 | 373.58 | 1.983 |
| point_014 | 380.32 | 362.05 | 5.0% | 739.67 | 377.62 | 2.043 |
| bridge_dense_nvfp4_072 | 707.11 | 647.30 | 9.2% | 902.92 | 255.62 | 1.395 |
| bridge_dense_nvfp4_088 | 672.41 | 622.32 | 8.0% | 858.34 | 236.02 | 1.379 |
| bridge_dense_nvfp4_104 | 557.91 | 523.74 | 6.5% | 742.66 | 218.91 | 1.418 |
| bridge_dense_nvfp4_120 | 531.50 | 505.60 | 5.1% | 705.21 | 199.60 | 1.395 |

Interpretation: predictor-vs-exact local error diagnoses the kernel model; exact-local-vs-E2E residual diagnoses model/runtime composition.

## Same-runtime phase-vLLM `apply()` validation

The preceding E2E column comes from the historical closure runner, not from an
instrumented measurement of the identical phase-vLLM execution.  To test the
central hypothesis directly, `profile_prefill_vllm_apply.py` monkey-patches
`PhaseHeteroMyTestLinearMethod.apply()` and records CUDA events around every
linear call.  Both policies are exported through the same
`export_phase_checkpoint.py` path and run with B=8, L=2048, eager execution,
chunked prefill disabled, and two warmed passes.

| policy | standalone exact local (ms) | phase-vLLM `apply` sum (ms) | phase-vLLM E2E wall (ms) | agreement |
|---|---:|---:|---:|---|
| point_000 (dense BF16) | 968.75 | 966.36 / 981.50 | 1089.72 / 1106.29 | within 1.3% |
| point_014 (high-sparse mixed) | 362.05 | 357.46 / 358.59 | 479.70 / 480.69 | within 1.3% |

This falsifies the earlier explanation that the high-sparse anomaly is caused
by a large wrapper-versus-phase-vLLM linear-path gap.  The same-runtime linear
sum agrees with the standalone exact-module measurement for both endpoints.
The old closure values (1294.20 ms for p000 and 739.67 ms for p014) therefore
cannot be used as the reference for this runtime decomposition: their
measurement boundary, execution configuration, or external interference must
differ.  They are retained only as historical artifacts and must not be used
to refit the local roofline predictor.

## Controlled scheduler and warm-state A/B

The historical closure benchmark is reproducible (p000 again measured
1293.11 ms), but it launches a fresh vLLM process for every nominal
"warmup"/sample and leaves `max_num_batched_tokens` at the runtime default.
It is therefore cold for every sample and may schedule the B=8, L=2048
workload in more than one prefill wave.  The controlled runner keeps the same
fresh-process `generate()` timing boundary and changes only the cap to 16384.

| policy | historical closure (ms) | cold, cap=16384 (ms) | warmed, cap=16384 (ms) |
|---|---:|---:|---:|
| point_000 | 1294.20 | 1149.07 | 1089.72 / 1106.29 |
| point_014 | 739.67 | 529.16 | 479.70 / 480.69 |

Thus the apparent speed-model error is a timing-protocol artifact, not a
failure of the local roofline-plus-calibration model: the historical protocol
adds a policy-dependent scheduler/cold-start component (145 ms for p000 and
211 ms for p014 before the additional warmed-state reduction).  All future
Llama3 prefill E2E measurements must use a single loaded vLLM engine, one
unrecorded request warmup, `max_num_batched_tokens=16384`,
`max_num_seqs=8`, eager mode, disabled chunked prefill and prefix cache, then
five timed requests.  The existing historical Llama3 prefill speed table and
its E2E calibrator should be regenerated under this protocol; the canonical
weights, quality measurements, policy solver, and local latency predictor are
reusable.
