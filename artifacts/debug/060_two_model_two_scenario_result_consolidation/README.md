# Two-model, two-scenario consolidated experiment bundle

This directory consolidates the retained results from four canonical experiment tracks. It does not rerun, alter, or replace any source experiment.

- [Llama2-7B-Chat prefill-only](llama2_7b_chat/prefill_only/summary.md)
- [Llama2-7B-Chat prefill-decode](llama2_7b_chat/prefill_decode/summary.md)
- [Llama3.1-8B-Instruct prefill-only](llama31_8b_instruct/prefill_only/summary.md)
- [Llama3.1-8B-Instruct prefill-decode](llama31_8b_instruct/prefill_decode/summary.md)

Every scenario contains `data/`, `policies/`, `results/`, `figures/`, and `summary.md`. Checkpoints and large raw logs are intentionally not duplicated; source paths are recorded in each summary.
