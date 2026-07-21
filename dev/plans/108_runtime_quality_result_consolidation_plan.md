# Runtime-quality result consolidation plan

Create a new, read-only debug-045 bundle that joins existing measured speed
artifacts with the corrected real-vLLM quality artifacts for Llama2-7B-Chat
and Llama3.1-8B-Instruct. Existing exports and historical summaries remain
unchanged.

1. Inventory the source schemas and policy aliases → verify each row has an
   unambiguous speed and corrected quality source.
2. Build a reproducible merger for prefill-only five-task metrics and
   prefill-decoding NLL → verify no field is taken from the invalid proxy
   quality outputs.
3. Generate Markdown/CSV tables and speed-vs-quality plots under debug 045 →
   verify row coverage and clearly label missing downstream generation metrics
   as historical rather than corrected.
