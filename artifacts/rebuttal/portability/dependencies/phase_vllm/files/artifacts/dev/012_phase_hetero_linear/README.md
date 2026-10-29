# Phase-Heterogeneous Linear Prototype

This artifact generalizes the previous fixed `dense_nvfp4 -> w4a16_ours`
phase switch into an artifact-local vLLM quantization backend named
`phase_hetero_mytest`.

Supported per-phase Linear methods:

- `dense_bf16`
- `sparse_bf16`
- `dense_nvfp4`
- `sparse_nvfp4`
- `w4a16_ours`

The exported checkpoint stores prefill weights under `ph_prefill_*`. If the
decode method differs, decode weights are stored under `ph_decode_*` and loaded
into CPU pinned memory at model load. During offline generation,
`enable_phase_hetero()` starts prefill mode; after each Linear finishes prefill,
that layer asynchronously copies its decode weights to GPU. Same-method layers
reuse the prefill weights and do not copy.

The implementation is intentionally artifact-local so it can be validated before
moving into the main vLLM quantization package.
