# Llama-3.1-8B-Instruct Baseline Implementation Log

## 2026-07-10 - Resumable quality generation
- Development purpose: preserve completed PMPD generations across interrupted GPU jobs.
- Changes: added opt-in `--resume` with run-config and JSONL-prefix validation, partial-final-line repair, per-record flushing, and no-model-load handling for complete files; enabled it in the batch quality runner.
- Affected files: `artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/pmpd_vllm_eval.py`, `run_all_quality.sh`, and `README.md`.
- Follow-up: partial Llama-3.1 quality JSONLs can now continue from their existing row counts; sparse BF16 jobs should retain `--gpu-memory-utilization 0.75`.

## 2026-07-10 - Dataset-specific resume validation
- Development purpose: allow CNN/DM and DialogSum runs to resume after the IWSLT-only filter tokenizer default changed.
- Changes: validate `iwslt_filter_tokenizer` only for IWSLT; all generation-affecting common fields remain strict.
- Affected files: `artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/pmpd_vllm_eval.py`.

## 2026-07-10 - Complete baseline summary
- Development purpose: publish the completed Llama-3.1-8B-Instruct baseline measurements in the same format as Llama2-7B-Chat.
- Changes: merged all five method speed outputs and generated the final Markdown/CSV summary with 10 speed rows and 15 quality rows.
- Affected files: `artifacts/exports/vllm/baselines/llama3.1-8b-instruct/results/summary/`.
- Follow-up: CNN/DM is the fixed 1000-example subset; IWSLT uses the Llama2 tokenizer fallback and the PMPD Claude-style prompt for cross-model comparability.
