# llama2-7b-chat: prefill-only multi-task quality (full)

| family | policy | recommended_use | e2e_ms | speedup | wikitext_word_ppl | winogrande_acc | arc_easy_acc | arc_easy_acc_norm | mmlu_acc | status |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | dense_bf16 | baseline | 1079.54 | 1 | 11.6025 | 0.683504 | 0.72601 | 0.683923 | 0.464321 | complete |
| ours | ours_point_004 | — | 1062.2 | 1.016 | 11.6534 | 0.681137 | 0.724747 | 0.683081 | 0.464891 | complete |
| uniform | marlin_nvfp4 | — | 1034.99 | 1.043 | 12.0444 | 0.678769 | 0.722643 | 0.671717 | 0.462541 | complete |
| ours | ours_point_006 | — | 1005.59 | 1.074 | 11.6714 | 0.678769 | 0.725168 | 0.684343 | 0.462541 | complete |
| ours | ours_point_008 | recommended: high-quality | 890.02 | 1.213 | 11.7036 | 0.686661 | 0.724747 | 0.680556 | 0.464891 | complete |
| ours | ours_point_009 | — | 791.93 | 1.363 | 11.8112 | 0.682715 | 0.72601 | 0.676768 | 0.457983 | complete |
| uniform | sparse_bf16 | — | 660.95 | 1.633 | 18.7154 | 0.639305 | 0.637205 | 0.582912 | 0.283008 | complete |
| ours | ours_point_011 | — | 660.34 | 1.635 | 12.0909 | 0.684294 | 0.721801 | 0.678451 | 0.455491 | complete |
| ours | ours_point_012 | recommended: primary balanced | 625.45 | 1.726 | 13.1766 | 0.668508 | 0.724747 | 0.685185 | 0.444808 | complete |
| ours | ours_point_013 | recommended: dense-NVFP4 cover | 596 | 1.811 | 13.764 | 0.659826 | 0.704125 | 0.676347 | 0.425509 | complete |
| uniform | dense_nvfp4 | — | 578.16 | 1.867 | 12.0444 | 0.678769 | 0.722643 | 0.671717 | 0.462541 | complete |
| ours | ours_point_015 | — | 552.74 | 1.953 | 23.4008 | 0.595107 | 0.600589 | 0.555135 | 0.257656 | complete |
| ours | ours_point_016 | recommended: max-speed endpoint | 528.44 | 2.043 | 30.1147 | 0.574586 | 0.545455 | 0.496633 | 0.23508 | complete |
| uniform | sparse_nvfp4 | — | 520.63 | 2.074 | 44.3065 | 0.543015 | 0.399832 | 0.377946 | 0.229739 | complete |
