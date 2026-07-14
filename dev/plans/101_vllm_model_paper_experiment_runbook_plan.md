# vLLM model paper-experiment runbook

## Goal

Write a handoff-ready, end-to-end runbook under `artifacts/exports/vllm/` for
reproducing the paper experiment of one model, using Llama2-7B-Chat as the
concrete example. It must explain the full workflow rather than merely listing
commands: environment/backend checks, uniform baselines, modeling, Pareto
search, measured closure, downstream validation, and paper-result packaging.

## Plan

1. Map each experiment stage to the currently successful Llama2 scripts and
   artifacts. → verify all referenced inputs/outputs exist in this workspace.
2. Write the runbook with command templates, fixed workload definitions,
   decision/checkpoint gates, and provenance rules for speed/quality results.
   → distinguish native uniform quant-method benchmarks from phase-hetero ours
   checkpoints and distinguish predicted from measured curves.
3. Include a migration checklist for a new machine/model and an explicit list
   of items that must be re-profiled rather than copied. → verify the final
   document links its result-bundle endpoint and primary scripts.
4. Record the implementation in `dev/impls/101_vllm_model_paper_experiment_runbook_impl.md`.
