# All-task-metrics result bundle

## Goal

Expand both final `artifacts/exports/vllm/ours/{llama2-7b-chat,llama3.1-8b-instruct}/pareto_summary/` bundles so their tables retain every measured downstream metric and their figures include one Pareto plot per metric.

## Plan

1. Locate and validate raw CNN/DM, DialogSum and IWSLT `metrics.json` files for uniform and all published ours points. → verify one consistent metric record per table row.
2. Extend both aggregation scripts and CSV/Markdown schemas with CNN/DM BERTScore, DialogSum BERTScore and IWSLT ROUGE-L. → preserve existing primary-metric columns and all speed/NLL provenance.
3. Generate three additional figures per model for those metrics, retaining existing primary figures. → verify the expected total figure set and that only rows with measured metrics are plotted.
4. Record the implementation and report the updated result directories.
