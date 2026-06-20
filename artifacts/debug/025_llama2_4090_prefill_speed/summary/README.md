# Llama2-7B 4090 Prefill/Decode Speed Summary

- Model path: `/data/home/scxj523/run/wja/data/models/LLM-Research/llama-2-7b`
- Scenarios:
  - `decode_heavy`: `batch_size=1, input_tokens=1024, output_tokens=256, m_prefill=1024, m_decode=1`
  - `prefill_decode`: `batch_size=1, input_tokens=16384, output_tokens=32, m_prefill=16384, m_decode=1`
- Methods: `dense_bf16, sparse_bf16, marlin_nvfp4`
- Full-model rows: `18`
- Linear aggregate rows: `48`

Primary result: `results/full_model_summary.csv`.
