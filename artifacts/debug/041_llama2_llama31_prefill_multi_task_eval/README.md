# Prefill-only multi-task quality closure

This debug-only experiment extends the existing prefill-only ARC-Challenge
closure to WikiText perplexity, Winogrande, ARC-Easy, and MMLU for both
Llama2-7B-Chat and Llama-3.1-8B-Instruct.  C4 is intentionally excluded.

Accuracy is evaluated with the same eager Transformers + `lm_eval` HFLM
backend used by `040_llama2_7b_dense_prefill_eval`; it is not a vLLM
throughput measurement.  Existing vLLM prefill-only speed results are read
from the paper-summary tables and joined only in this directory's summaries.
Llama2 WikiText uses the HFLM default context; Llama3.1 uses a fixed 1024-token
context because eager 2048-token attention exceeds the 32GB RTX 5090 memory.

## Prepare and validate

```bash
conda run -n cospaq python scripts/build_manifest.py
conda run -n cospaq python scripts/summarize.py
```

The manifest covers every prefill-only uniform and `ours_point_*` row in the
two existing `pareto_summary/summary.md` files: 14 Llama2 policies and 12
Llama3.1 policies.

## Run when GPUs are available

First run the six-policy smoke set (four tasks each):

```bash
conda run -n cospaq python scripts/run_all.py --selection \
  llama2-7b-chat:dense_bf16,llama2-7b-chat:dense_nvfp4,llama2-7b-chat:ours_point_012,\
llama3.1-8b-instruct:dense_bf16,llama3.1-8b-instruct:dense_nvfp4,llama3.1-8b-instruct:ours_point_6
```

After reviewing the smoke results, run the remaining policies (the runner
skips valid completed results by default):

```bash
conda run -n cospaq python scripts/run_all.py --gpus 1,2,3,4,5,6,7
conda run -n cospaq python scripts/summarize.py
```

Use `--limit N` only for a diagnostic profile.  Full outputs are under
`results/<model>/<policy>/<task>/full/result.json`; no file below
`artifacts/exports` is modified.
