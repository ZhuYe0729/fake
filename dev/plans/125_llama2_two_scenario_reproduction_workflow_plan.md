# 125 Llama2 two-scenario reproduction workflow plan

## Goal

Create a concrete, command-level reference workflow under a new debug bundle
that reproduces the final Llama2-7B-Chat prefill-only and prefill-decode paper
experiments.  The workflow must be usable by a later agent on another machine,
while being exercised end-to-end on the current machine with the `cospaq` and
`vllm` conda environments.

## Decisions

- The guide bundle is
  `artifacts/debug/063_llama2_two_scenario_reproduction_workflow/`.
- The retained source protocols are 054 prefill-only and 056
  B=8/S=2048/O=64 prefill-decode; 060 is the reference consolidation schema.
- Main-paper uniform and mixed policies use the same
  `phase_hetero_mytest` runtime and canonical sparse states.
- Machine-specific paths are centralized; old entrypoint defaults remain
  backward compatible.
- Validation combines fresh four-GPU smoke measurements with deterministic
  replay/audit of the already completed full 054/056/060 artifacts.

## Acceptance criteria

- The runbook contains executable commands for environment checks, extension
  prewarm, canonical sparse preparation, baselines, quality/speed modeling,
  solving, closure, downstream tasks, and paper aggregation.
- Bootstrap and validation tools can target a new debug run directory without
  modifying 054, 056, or 060.
- Both conda environments are checked explicitly and used for their intended
  stages.
- A validation report distinguishes freshly executed smoke steps, full-data
  replay, and commands that were only statically checked.
- Known regressions are blocked: direct pruning, repeated NVFP4 packing,
  runtime mismatch, wrong B/O scenario, concurrent JIT compilation, and
  incomplete task-shard merges.
