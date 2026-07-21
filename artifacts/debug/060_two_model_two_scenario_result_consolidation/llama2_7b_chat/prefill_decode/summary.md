# llama2_7b_chat: prefill-decode

B=8, input=2048, output=64; phase-vLLM runtime, BF16 KV cache, chunked prefill disabled.

This is a compact consolidation of `artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/llama2_7b_chat`. No measurement was rerun. `data/complete_results.csv` is the machine-readable version of the full retained task-result table; empty E2E/TTFT fields mean the corresponding source recorded speedup but did not retain the raw latency in this table.

## Complete measured-result table

| policy_id | e2e_median_ms | ttft_median_ms | measured_speedup_vs_dense | raw_predicted_speedup_vs_dense | predicted_delta_nll | cnn_dm_rougeL_percent | cnn_dm_bert_score_percent | dsum_rougeL_percent | dsum_bert_score_percent | iwslt_rougeL_percent | iwslt_sacre_bleu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b8o64000 | 2305.1626067608595 | 1115.991149097681 | 1.0 | 1.0 | 0.0 | 23.61032474800552 | 87.18523595929146 | 21.598181660389766 | 87.15889004866281 | 46.908911718967225 | 19.411758651336964 |
| b8o64001 | 2357.8019496053457 | 1173.553243279457 | 0.9776744001533729 | 1.046344500220006 | 0.00048125893935449965 | — | — | — | — | — | — |
| b8o64002 | 2418.840691447258 | 1159.1685209423304 | 0.9530030708147291 | 1.0971933790151636 | 0.0009066674968803757 | — | — | — | — | — | — |
| b8o64003 | 2217.064144089818 | 1011.6227362304926 | 1.039736542086927 | 1.2153138374398254 | 0.0018632991789123028 | 23.9272874575251 | 87.22371274232864 | 21.55216240817209 | 87.17124141454697 | 47.31949094099562 | 19.75522973711901 |
| b8o64004 | 2008.4384083747864 | 1063.189823180437 | 1.147738759201573 | 1.4245514949465892 | 0.006002522399425643 | 23.798308546344927 | 87.23485642671585 | 21.400611305299233 | 86.99458617369335 | 47.78187497023512 | 21.083232116943357 |
| b8o64005 | 1938.7222286313772 | 869.5199992507696 | 1.1890112841942124 | 1.5893446942749225 | 0.01802637583983648 | 20.837734982657743 | 86.22640173435211 | 16.905878820964983 | 85.74235810041428 | 21.63043551878755 | 6.987439317418688 |
| b8o64006 | 1751.6449578106403 | 839.5041339099407 | 1.3159987681762029 | 1.9212939290276523 | 0.03646033170064827 | 19.266268656956594 | 85.56828852891923 | 15.867386181824235 | 85.01695797443391 | 18.801996961494645 | 7.151955584974809 |
| b8o64007 | 1599.069906398654 | 729.2589358985424 | 1.4415646229953964 | 2.280772787241016 | 0.074467094894349 | 16.66491149455221 | 84.39397824406623 | 14.18585913004713 | 84.03823717037837 | 17.967041086806237 | 4.008009470711757 |
| b8o64008 | 1557.1518409997225 | 673.3864843845367 | 1.4803711147917977 | 2.4603342644799677 | 0.1747028104033382 | 17.940389532035848 | 85.12067812681198 | 13.218303377594085 | 84.10107502937318 | 15.454714227052063 | 2.7903595157363323 |
| b8o64009 | 1477.0602080971003 | 731.9759894162416 | 1.5606422772234894 | 2.488567140038566 | 0.19306166188909027 | 18.562628851990738 | 85.21325730085373 | 13.573771019025024 | 84.24758412837983 | 15.569602767385286 | 2.6026263063506963 |

## Uniform baselines: corrected 056 source table

The rows below are explicit uniform baseline results from 056, not solver policies. `p03` is a legal projection: sparse NVFP4 in prefill and dense NVFP4 in decode, because sparse NVFP4 is unsupported for decode M=8. p03/p04 primary task scores were subsequently backfilled with the same phase-vLLM runtime and PMPD primary-metric definitions; only BERTScore remains intentionally unmeasured for those two rows.

| method | policy | E2E ms | TTFT ms | TPOT ms | E2E speedup | CNN/DM R-L | DialogSum R-L | IWSLT BLEU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense BF16 | p00 | 2291.84 | 1096.90 | 18.97 | 1.00x | 23.67 | 21.69 | 19.30 |
| dense NVFP4 | p01 | 2853.99 | 668.07 | 34.70 | 0.80x | 24.27 | 20.59 | 16.84 |
| sparse BF16 | p02 | 2020.77 | 701.85 | 20.94 | 1.13x | 15.35 | 13.54 | 3.90 |
| sparse NVFP4 legal projection | p03 | 2991.50 | 679.03 | 36.71 | 0.77x | 15.96 | 13.57 | 1.57 |
| W4A16 Marlin | p04 | 1925.83 | 1128.22 | 12.66 | 1.19x | 23.73 | 21.77 | 18.85 |

## Figures

- [figures/pareto_IWSLT.png](figures/pareto_IWSLT.png)
- [figures/pareto_cnn_dm_1000.png](figures/pareto_cnn_dm_1000.png)
- [figures/pareto_dsum.png](figures/pareto_dsum.png)
- [figures/pareto_IWSLT_with_uniform_056.png](figures/pareto_IWSLT_with_uniform_056.png)
- [figures/pareto_cnn_dm_1000_with_uniform_056.png](figures/pareto_cnn_dm_1000_with_uniform_056.png)
- [figures/pareto_dsum_with_uniform_056.png](figures/pareto_dsum_with_uniform_056.png)

## Caveat

The complete table is the measured downstream-task subset of the closure; `data/predicted_points.csv` retains the wider solver candidate set. Uniform speed is complete for p00--p04 in `data/uniform_baselines.csv`; the explicit uniform task table is `data/uniform_baseline_results.csv`. p00--p02 scores are retained original 056 scores; p03/p04 primary metrics are this backfill's real phase-vLLM results. Do not infer missing secondary metrics from another model or experiment.
