# Llama prefill-decoding real-vLLM NLL plan

Replace only the invalid prefill-decoding NLL proxy with new, isolated
real-vLLM measurements for Llama2-7B-Chat and Llama3.1-8B-Instruct. Existing
speed and generation-task artifacts remain immutable.

1. Audit the published prefill-decoding policy inventory and existing phase
   transition runner → verify the new NLL path triggers an actual prefill to
   decode boundary.
2. Implement isolated manifests, runtime checkpoint materialization, and a
   reproducible NLL runner → verify uniform quantization and phase-hetero
   methods are loaded by vLLM, not by a weight proxy.
3. Run uniform and published ours policies for both models, with sparse export
   using 2:4 pruning → verify every intended row has NLL and runtime metadata.
4. Write separate summaries that label old NLL as historical and preserve all
   existing result files.
