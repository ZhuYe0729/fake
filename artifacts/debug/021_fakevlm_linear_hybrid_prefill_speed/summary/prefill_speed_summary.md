# FakeVLM Prefill Speed Summary

| Batch | Family | Latency ms | Samples/s | Best uniform | Speedup |
|---:|---|---:|---:|---|---:|
| 1 | manual_profile | 64.374396 | 15.534126 | uniform_sparse_bf16 | 1.022259 |
| 1 | latency_model | 60.203033 | 16.610459 | uniform_sparse_bf16 | 1.093089 |
| 2 | manual_profile | 105.338689 | 18.986376 | uniform_sparse_nvfp4 | 0.980488 |
| 2 | latency_model | 100.744896 | 19.852122 | uniform_sparse_nvfp4 | 1.025196 |
| 4 | manual_profile | 192.355710 | 20.794808 | uniform_sparse_nvfp4 | 1.060813 |
| 4 | latency_model | 192.638408 | 20.764291 | uniform_sparse_nvfp4 | 1.059256 |
| 8 | manual_profile | 404.644479 | 19.770441 | uniform_sparse_bf16 | 1.070513 |
| 8 | latency_model | 405.282205 | 19.739332 | uniform_sparse_bf16 | 1.068829 |
| 16 | manual_profile | 804.428046 | 19.889908 | uniform_sparse_bf16 | 1.073394 |
| 16 | latency_model | 804.234747 | 19.894689 | uniform_sparse_bf16 | 1.073652 |
