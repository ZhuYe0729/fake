# llama3.1-8b-instruct: prefill-only multi-task quality (full)

| family | policy | recommended_use | e2e_ms | speedup | wikitext_word_ppl | winogrande_acc | arc_easy_acc | arc_easy_acc_norm | mmlu_acc | status |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | dense_bf16 | baseline | 1187.12 | 1 | 10.2696 | 0.734017 | 0.819865 | 0.797559 | 0.683735 | complete |
| uniform | dense_nvfp4 | — | 687.15 | 1.728 | 10.6605 | 0.737174 | 0.813973 | 0.791667 | 0.665931 | complete |
| uniform | sparse_bf16 | — | 766.07 | 1.55 | 23.1426 | 0.656669 | 0.677609 | 0.632155 | 0.397522 | complete |
| uniform | sparse_nvfp4 | — | 618.46 | 1.919 | 65.284 | 0.538279 | 0.484007 | 0.449916 | 0.237929 | complete |
| uniform | marlin_nvfp4 | — | 1156.28 | 1.027 | 10.6605 | 0.737174 | 0.813973 | 0.791667 | 0.665931 | complete |
| ours | ours_point_3 | — | 969.65 | 1.224 | 10.3811 | 0.744278 | 0.817761 | 0.787458 | 0.683165 | complete |
| ours | ours_point_5 | — | 826.11 | 1.437 | 10.4682 | 0.742699 | 0.808502 | 0.784933 | 0.680387 | complete |
| ours | ours_point_6 | recommended: near-lossless | 760.76 | 1.56 | 10.5238 | 0.738753 | 0.805976 | 0.784933 | 0.676043 | complete |
| ours | ours_point_8 | recommended: dense-NVFP4-cover | 663.98 | 1.788 | 12.1673 | 0.725335 | 0.801768 | 0.792508 | 0.616864 | complete |
| ours | ours_point_9 | — | 639.62 | 1.856 | 12.8427 | 0.725335 | 0.789983 | 0.758838 | 0.573992 | complete |
| ours | ours_point_11 | optional: high-speed trade-off | 583.41 | 2.035 | 15.4999 | 0.683504 | 0.742003 | 0.707912 | 0.443313 | complete |
| ours | ours_point_13 | — | 547.46 | 2.168 | 33.22 | 0.626677 | 0.614899 | 0.579545 | 0.306082 | complete |
