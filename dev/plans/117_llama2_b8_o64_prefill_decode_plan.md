# 117 Llama2 B=8/O=64 prefill-decode plan

## Goal

Rebuild the Llama2-7B-chat prefill-decode Pareto experiment for batch 8,
2048-token prompts, and 64 generated tokens.  The experiment uses BF16 KV
cache and a canonical sparse-weight runtime, avoiding V1 request-wave phase
errors, FP8 KV approximation, repeated quantization, and direct pruning.

## Decisions

- New artifacts live in `artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/`.
- Reuse the kernel predictor, canonical sparse states, policy coverage suite,
  and quality-model family; re-audit new kernel shapes and refit scenario-level
  speed and quality calibration.
- Formal speed protocol: V1, no chunked prefill, no prefix cache, BF16 KV,
  `gpu_memory_utilization=0.80`, and fresh vLLM process per timed request.
- PMPD remains a real-task quality validation at its existing batch-4,
  max-new-token-256 protocol.

## Acceptance criteria

- Dense and representative mixed policy traces show a single 16384-token
  prefill wave followed by decode batches of 8.
- All sparse exports have canonical provenance and `prune:false`.
- New action audit uses M=16384/M=8; speed calibration and quality calibration
  report held-out errors.
- Pareto closure uses measured B=8/O=64 speed and real O=64 NLL, with uniform
  methods plotted under the same runtime configuration.
