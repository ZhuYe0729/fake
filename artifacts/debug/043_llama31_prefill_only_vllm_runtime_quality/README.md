# Llama3.1-8B-Instruct prefill-only real-vLLM quality closure

This debug experiment replaces the invalid weight-proxy/HF quality results for
prefill-only inference with the actual vLLM runtime used by the speed runs.
It evaluates WikiText, Winogrande, ARC-Easy, ARC-Challenge, and MMLU at
zero-shot for the five uniform baselines and seven published heterogeneous
Pareto points.  Existing speed measurements are intentionally reused.

`scripts/build_manifest.py` makes the 12-policy inventory.  Uniform
checkpoints are reused directly; `scripts/materialize_all.py` exports only the
seven missing prefill-only phase-heterogeneous checkpoints into this directory.
`scripts/run_all.py --audit --limit 1` is the mandatory runtime/packing audit
before the full evaluation.  `scripts/summarize.py` joins quality with the
existing measured E2E speed.

Sparse BF16 quality evaluation sets a process-local cache limit of 16 for the
patched sparse wrapper.  The kernel default remains 512, and no speed artifact
is changed by this experiment.
