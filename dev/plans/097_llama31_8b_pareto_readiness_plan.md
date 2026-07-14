# Llama-3.1-8B-Instruct Pareto Readiness Plan

## Objective

Convert the validated Llama2 Pareto workflow into a model-transfer playbook for
Llama-3.1-8B-Instruct. This plan prepares the next experiment; it does not
claim that calibration or policies transfer across models.

## Deliverables

1. A final Llama2 status summary with measured evidence, valid claims, and
   known limitations for both serving scenarios.
2. A gated Llama3.1 execution guide: architecture audit, fresh calibration,
   speed/quality validation, discrete solving, and task-level reporting.
3. An explicit file map and failure-recovery checklist so the next run does not
   rediscover runner, GPU-memory, or baseline-comparison issues.

## Verification

- Every recommended step has an input, output, acceptance gate, and source
  script/artifact path.
- The guide states that Llama3.1 needs fresh quality and E2E calibration,
  including its GQA fused-QKV shape.
- The guide requires a union Pareto computation with uniform references and
  measured axes before any figure is called final.
