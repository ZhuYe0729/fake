# Prefill-decode PMPD task-quality validation

All speed values use the `.85` formal protocol; task metrics use isolated fresh-process vLLM generation.

| point | dataset | speedup | WikiText ΔNLL | ROUGE-L | BERTScore | SacreBLEU | empty |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | cnn_dm_1000 | 1.010 | 0.073 | 23.623 | 87.178 |  | 0 |
| 2 | dsum | 1.010 | 0.073 | 21.760 | 87.176 |  | 0 |
| 2 | IWSLT | 1.010 | 0.073 | 47.087 |  | 19.565 | 0 |
| 7 | cnn_dm_1000 | 1.165 | 0.639 | 23.648 | 87.168 |  | 0 |
| 7 | dsum | 1.165 | 0.639 | 21.863 | 87.200 |  | 0 |
| 7 | IWSLT | 1.165 | 0.639 | 46.198 |  | 19.256 | 0 |
| 9 | cnn_dm_1000 | 1.071 | 1.092 | 23.770 | 87.201 |  | 0 |
| 9 | dsum | 1.071 | 1.092 | 21.419 | 87.156 |  | 0 |
| 9 | IWSLT | 1.071 | 1.092 | 44.423 |  | 18.419 | 0 |
| 10 | cnn_dm_1000 | 1.356 | 1.819 | 23.424 | 87.055 |  | 0 |
| 10 | dsum | 1.356 | 1.819 | 21.300 | 87.135 |  | 0 |
| 10 | IWSLT | 1.356 | 1.819 | 43.473 |  | 16.816 | 0 |
