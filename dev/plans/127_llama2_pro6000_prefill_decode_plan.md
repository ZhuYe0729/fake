# 127 Llama2-7B-Chat / Pro 6000 / Prefill-Decode

## Goal

Build an isolated, restartable B=8/S=2048/O=64 prefill-decode paper workflow in
`artifacts/debug/065_llama2_pro6000_prefill_decode`.  The frozen protocol follows
063, while the implementation carries forward the Pro 6000 guards validated by 064.

## Decisions

- All experiment-owned code, policies, canonical copies, profiles, logs and results live in 065.
- Copy and SHA-256 audit the two canonical states from 064; never direct-prune or requantize the sparse-NVFP4 source twice.
- Fit 72 teacher-forced policies with B=8/S=2048/O=64 and a 16896-token scheduler budget.
- Profile M=16384 and M=8 locally, exclude sparse-NVFP4 decode before solving, and close every unique candidate with measured 1+5 E2E speed.
- Run fixed PMPD CNN/DM-1000, DialogSum-1500 and IWSLT-333 on five uniform and measured-frontier policies.

## Acceptance

`validate all` must prove isolation, canonical provenance, 72/72 NLL, valid decode traces,
dual-shape profiling, complete closure measurements and exact downstream question coverage.
