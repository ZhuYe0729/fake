# Two-model, two-scenario experimental result consolidation plan

## Goal

Create one new debug artifact that consolidates the currently available paper-experiment outputs for Llama2-7B-Chat and Llama3.1-8B-Instruct. Each model will have independent prefill-only and prefill-decode subdirectories containing policies, result data, figures, provenance, and a complete summary table.

## Scope and assumptions

- This is a read-only consolidation of existing experiments; no GPU benchmark, export, re-evaluation, or model/runtime change is allowed.
- The bundle must preserve source paths and distinguish canonical/current results from legacy or known-diagnostic outputs.
- "Complete" means every policy with an existing measured speed/quality result is tabulated. Missing fields must be shown as missing, not inferred.

## Steps

1. Inventory existing Llama2/Llama3 artifacts for both scenarios and select the latest internally consistent source table, policies, and figures.
   - Verify: each selected source has an explicit provenance record.
2. Create `artifacts/debug/060_two_model_two_scenario_result_consolidation/` with `llama2_7b_chat/` and `llama31_8b_instruct/`, each split into `prefill_only/` and `prefill_decode/`.
   - Verify: every scenario includes `data/`, `policies/`, `figures/`, and `summary.md`.
3. Copy only compact derived data, policy JSONs, relevant plots, and source/provenance manifests; do not duplicate checkpoints or large raw logs.
   - Verify: the consolidated artifact remains self-contained for reading and reproducible through recorded source paths.
4. Generate complete Markdown/CSV tables from selected source summaries and clearly annotate measurement runtime, scenario, metric definitions, and known caveats.
   - Verify: the Markdown and CSV row counts agree and all copied figures are linked.
