# Llama3 prefill speed protocol and bug record

## The bug

The old Llama3 prefill closure measured each nominal repeat in a newly loaded
vLLM process, with no request warmup, and did not explicitly set
`max_num_batched_tokens`. Its reported E2E latency therefore included
first-request setup and a scheduler-cap-dependent number of prefill waves.
Those costs varied by policy and cannot be learned from the sum of Linear
latencies.

This is not a CUTLASS roofline-model failure. On the same exported checkpoints,
the sum of real phase-vLLM Linear `apply()` CUDA events matched standalone exact
module timing within 1.3% for both dense BF16 and a high-sparse mixed policy.

## Mandatory protocol

| setting | required value |
|---|---|
| scenario | prefill-only, B=8, input=2048, output=1 |
| runtime | `phase_hetero_mytest` for every policy/baseline |
| scheduling | `max_num_seqs=8`, `max_num_batched_tokens=16384` |
| execution | eager; prefix caching off; chunked prefill off |
| phase | initial warmup uses the runtime's initial prefill state; explicitly call `prepare_next_prefill()` and `wait_for_prefill_ready()` before every subsequent request |
| warmup | one unrecorded request in the same loaded vLLM engine |
| measurement | five subsequent requests in the same engine; report their median and raw values |

## Required JSON checks

Before a timing file can be used as a speed-calibration or closure label, it
must contain:

- `max_num_batched_tokens: 16384` and `max_num_seqs: 8`;
- exactly one warmup request and five `timed_ms` values;
- `protocol: same_engine_warmup_then_timed_explicit_prefill_phase`.

Do not mix historical 058 timing JSONs with these labels. They may be retained
for provenance but are invalid targets for the corrected E2E calibrator.

## Extension-cache isolation

The `cospaq` exporter and `vllm` runner must not share the dense-NVFP4
`torch.utils.cpp_extension` cache: they use different Python/PyTorch runtime
stacks. 061 sets `CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR` to its vLLM-specific
extension directory for every benchmark. A shared cache previously caused both
packing and dense-NVFP4 phase execution to stall.

The same separation applies to sparse-NVFP4 and sparse-BF16 extensions. The
exporter receives `cospaq_*` directories and the benchmark receives `vllm_*`
directories; they must never point to one shared `torch_extensions` directory.
