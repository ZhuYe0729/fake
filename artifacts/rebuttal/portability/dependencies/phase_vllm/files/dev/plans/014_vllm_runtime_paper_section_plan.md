# vLLM Runtime Paper Section Plan

## Goal
- Draft a focused paper-ready LaTeX section explaining the vLLM implementation
  of phase-wise heterogeneous policies and its runtime ablation.

## Approach
- Describe policy-driven per-Linear routing, asynchronous decode-weight
  materialization on a copy stream, CUDA-event dependencies, and prefill-weight
  release.
- Provide one compact runtime table that isolates synchronous switching,
  asynchronous overlap, and the memory lifecycle choice without inventing
  experimental values.
