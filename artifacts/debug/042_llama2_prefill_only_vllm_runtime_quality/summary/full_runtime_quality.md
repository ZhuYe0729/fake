# Llama2 prefill-only: actual vLLM runtime quality

| family | policy | recommended_use | e2e_ms | speedup | wikitext_word_ppl | winogrande_acc | arc_easy_acc | arc_easy_acc_norm | arc_challenge_acc | arc_challenge_acc_norm | mmlu_acc | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | dense_bf16 | baseline | 1079.54 | 1 | — | — | — | — | — | — | — | pending |
| ours | ours_point_004 | — | 1062.2 | 1.016 | — | — | — | — | — | — | — | pending |
| uniform | marlin_nvfp4 | — | 1034.99 | 1.043 | — | — | — | — | — | — | — | pending |
| ours | ours_point_006 | — | 1005.59 | 1.074 | — | — | — | — | — | — | — | pending |
| ours | ours_point_008 | recommended: high-quality | 890.02 | 1.213 | — | — | — | — | — | — | — | pending |
| ours | ours_point_009 | — | 791.93 | 1.363 | — | — | — | — | — | — | — | pending |
| uniform | sparse_bf16 | — | 660.95 | 1.633 | — | — | — | — | — | — | — | pending |
| ours | ours_point_011 | — | 660.34 | 1.635 | — | — | — | — | — | — | — | pending |
| ours | ours_point_012 | recommended: primary balanced | 625.45 | 1.726 | — | — | — | — | — | — | — | pending |
| ours | ours_point_013 | recommended: dense-NVFP4 cover | 596 | 1.811 | — | — | — | — | — | — | — | pending |
| uniform | dense_nvfp4 | — | 578.16 | 1.867 | — | — | — | — | — | — | — | pending |
| ours | ours_point_015 | — | 552.74 | 1.953 | — | — | — | — | — | — | — | pending |
| ours | ours_point_016 | recommended: max-speed endpoint | 528.44 | 2.043 | — | — | — | — | — | — | — | pending |
| uniform | sparse_nvfp4 | — | 520.63 | 2.074 | — | — | — | — | — | — | — | pending |
