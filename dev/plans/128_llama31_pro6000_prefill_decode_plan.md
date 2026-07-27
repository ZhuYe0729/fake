# 128 Llama3.1-8B-Instruct / Pro 6000 / Prefill-Decode

## Goal

Build an isolated, restartable B=8/S=2048/O=64 Llama3.1-8B-Instruct
prefill-decode paper workflow in
`artifacts/debug/066_llama31_pro6000_prefill_decode`.

## Decisions

- Keep every experiment-owned script, policy, canonical state, profile, log and result in 066; do not modify older experiment directories.
- Regenerate Pro 6000 Sparse-BF16 2:4 and sparse-NVFP4 pairwise 4:8 prequant-only canonical states because the historical large state files are unavailable locally.
- Freeze the historical 72-policy Llama3 design and deterministic 300×2112 Llama3-tokenized WikiText samples; formally score 100 blocks with B=8 teacher-forced 64-token continuation NLL.
- Profile exact Llama3 GQA/MLP shapes at M=16384 and M=8, exclude decode sparse-NVFP4, solve quality budgets, and close every unique point with measured single-process 1+5 E2E speed.
- Run fixed PMPD CNN/DM-1000, DialogSum-1500 and IWSLT-333 using the historical Legacy/common prompt protocol selected by the user.

## Acceptance

`validate all` must prove isolation, regenerated canonical provenance, 72/72 NLL,
valid phase traces, eight exact profile shapes, complete closure measurements,
Legacy task provenance and exact downstream question coverage.
