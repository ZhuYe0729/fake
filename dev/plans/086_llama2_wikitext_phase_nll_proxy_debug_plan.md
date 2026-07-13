# 086 Llama2 WikiText Phase-NLL Proxy Debug Plan

## Objective
- Isolate the quality-model investigation from exported Pareto artifacts.
- Validate normalized, low-dimensional, phase-separated NLL proxies on WikiText before any Pareto solve.

## Decisions
- Root: `artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/`.
- 72 deterministic controlled policies, fixed 54 train / 18 holdout split.
- WikiText-2 provides 300 offline blocks of 2048 prefill plus 80 decode tokens.
- Compare legacy, normalized-bucket, phase-separated, and phase-local-error variants.
- PMPD is limited to six post-fit transfer-only policies and never used for fitting.

## Verification
- Check policy coverage and decode legality before GPU work.
- Write per-policy shards and resumable 8-GPU workers.
- Report train/holdout MAE, RMSE, and Spearman for every ablation.
- Stop at model validation; Pareto solving remains TODO.
