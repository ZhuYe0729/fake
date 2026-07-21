# Llama-3.1-8B-Instruct B=8/O=64 canonical prefill-decode plan

## Objective

Create a new isolated `artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/` experiment and reproduce the completed Llama2-7B-chat prefill-decode closure for Meta-Llama-3.1-8B-Instruct:

- formal B=8, input=2048, output=64, BF16-KV, non-chunked phase runtime;
- canonical SparseGPT sparse-BF16/NVFP4 exports and audit;
- actual vLLM B=8 teacher-forced NLL labels, phase-aware quality fit, and constrained Pareto solve;
- fresh-process common-runtime speed closure;
- CNN/DM, DialogSum, IWSLT task closure, tables and Pareto figures.

## Plan

1. Bootstrap the 057 layout from the validated 056 scripts, replacing only model-specific configuration and paths. Verify model dimensions, sample tokenization, sparse-state provenance, and decode-kernel legality. → verify: scenario/action audit reports and one canonical export trace.
2. Collect local phase/method errors and B=8 real-vLLM NLL labels; fit the established phase-aware ReLU quality model. → verify: 72 labels with B=8/0.8/complete-wave metadata; held-out metrics report.
3. Solve legal candidate policies from the existing roofline action model and run common-runtime 5-repeat fresh-process speed closure. → verify: candidate policies pass canonical provenance and have measured E2E median speed.
4. Evaluate speed-relevant candidates on CNN/DM-1000, DialogSum-1500, IWSLT-333. Use B=1 retry tails only for OOM recovery, retaining all existing samples and recording the exception. → verify: all question ids complete and unique; 24 metrics files.
5. Produce result tables, speed/NLL and task Pareto plots, then audit counts/provenance before publishing the 057 report. → verify: report links, summary CSVs, figures, and implementation record.

## Resource gate

Llama-3.1 8B candidate checkpoints are roughly 15--16 GiB each. The current 75 GiB free disk space is insufficient to retain all 10 candidate checkpoints plus canonical states and task artifacts. Bootstrap/modeling can proceed, but before parallel candidate speed/task closure either reclaim at least 120 GiB or implement a sequential export-measure-task-clean workflow that preserves only policy JSON, provenance, measurements, and task outputs.
