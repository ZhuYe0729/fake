# Prefill-decode PMPD task-quality validation

All speed values use the `.85` formal protocol; task metrics use isolated fresh-process vLLM generation.

| point | dataset | speedup | WikiText ΔNLL | ROUGE-L | BERTScore | SacreBLEU | empty |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | cnn_dm_1000 | 1.000 | 0.000 | 23.610 | 87.186 |  | 0 |
| 0 | dsum | 1.000 | 0.000 | 21.598 | 87.161 |  | 0 |
| 0 | IWSLT | 1.000 | 0.000 | 46.909 |  | 19.412 | 0 |
| 3 | cnn_dm_1000 | 1.032 | 0.067 | 23.776 | 87.198 |  | 0 |
| 3 | dsum | 1.032 | 0.067 | 21.385 | 87.148 |  | 0 |
| 3 | IWSLT | 1.032 | 0.067 | 47.773 |  | 20.238 | 0 |
| 8 | cnn_dm_1000 | 1.179 | 0.715 | 23.819 | 87.188 |  | 0 |
| 8 | dsum | 1.179 | 0.715 | 21.643 | 87.105 |  | 0 |
| 8 | IWSLT | 1.179 | 0.715 | 45.432 |  | 18.920 | 0 |
| 11 | cnn_dm_1000 | 1.714 | 2.115 | 23.544 | 87.082 |  | 0 |
| 11 | dsum | 1.714 | 2.115 | 21.581 | 87.154 |  | 0 |
| 11 | IWSLT | 1.714 | 2.115 | 45.309 |  | 18.301 | 0 |
