# Llama-3.1-8B-Instruct canonical prefill-decode Pareto experiment

Formal protocol: batch 8, prompt 2048 tokens, output 64 tokens, BF16 KV cache,
unchunked prefill, and the phase-heterogeneous vLLM runtime for both uniform and
mixed policies.  The experiment deliberately starts clean because old `039` used
the pre-canonical sparse workflow and B=16/O=80.

`scripts/bootstrap_policies.py` installs the fixed 72-policy design with hashes.
`scripts/prepare_canonical_sparse.py` creates calibrated SparseGPT states; the
NVFP4 state is intentionally *not* pre-quantized, so the phase exporter packs it
exactly once.  `scripts/run_stage.sh` routes the validated 056 implementation to
these Llama-3 paths via explicit environment variables.

Disk policy: materialize/measure one phase checkpoint at a time and remove only
that temporary checkpoint after its speed or task artifact is complete.  Raw NLL,
speed summaries, policy manifests, and downstream task outputs are retained.
