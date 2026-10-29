# Canonical paper tables

- `main_table_final.tex`: Table 1, the complete RTX 5090 results for both
  models and both workloads.
- `cross_platform_pro6000_final.tex`: Table 2, the compact Llama-2-7B-Chat
  RTX PRO 6000 result used in the final paper.
- `ablation.tex`: the final modeling and runtime ablation table.

Only files in this directory are canonical.  Superseded tables are isolated
under `results/legacy_tables/` and are not read by validation or plotting
workflows.

`main_table_sources.csv` maps every RTX 5090 table row to its prefill,
generation-quality, speed, and BF16 reference source. The PRO 6000 and ablation
tables are generated directly from `selected_results.csv` and
`key_metrics.csv`. Rebuild all three tables with:

```bash
python workflows/paper/build_tables.py
```
