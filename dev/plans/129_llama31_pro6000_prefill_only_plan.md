# Llama3.1-8B-Instruct / Pro 6000 / Prefill-only

## Goal

Build an isolated, restartable B=8/S=2048/O=1 Llama3.1-8B-Instruct
prefill-only paper workflow in
`artifacts/debug/067_llama31_pro6000_prefill_only`.

## Decisions

- Keep every experiment-owned script, policy, intermediate, log and result in 067; historical debug directories remain read-only.
- Use the 72-policy Llama3 prefill-only design, regenerate a deterministic 100x2049 WikiText sample, and collect 100-block real-vLLM NLL labels.
- Copy the two canonical Llama3 sparse states from 066 into 067, then revalidate their hashes, 224-module coverage, structured sparsity and sparse-NVFP4 prequant-only provenance.
- Re-profile the four exact Llama3 prefill shapes on the current Pro 6000 and close every unique solver point with single-process 1+5 measured E2E timing.
- Evaluate the measured-frontier selection on WikiText, WinoGrande, ARC-Easy, ARC-Challenge and MMLU.

## Acceptance

`validate all` must prove isolation, 72/72 NLL coverage, canonical and checkpoint
provenance, four exact hardware-profile shapes, complete measured closure, five-task
coverage, and final tables/figures built only from 067 formal outputs.
