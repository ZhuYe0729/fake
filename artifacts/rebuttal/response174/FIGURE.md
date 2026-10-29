# Six-workload figure

Fixed Qwen3 policies D1 (`w1_e0.01`) and D32 (`w32_e0.03`), with no per-workload retuning. All six predeclared run040 cases are shown. Markers are the medians of three per-round speed ratios to INT8; whiskers are the full three-round minimum/maximum, not confidence intervals. End-to-end includes the prefill-to-decode transition; full cycle additionally includes restoration for the next prefill. This is a comparison to INT8, not a claim of dominance over all baselines or budget satisfaction at every workload. Some cycle ranges include parity even where the median is above one.

[PDF](qwen_six_workloads.pdf) · [SVG](qwen_six_workloads.svg) · [PNG](qwen_six_workloads.png) · [exact plotted data](workload_figure_data.csv) · [source](../runs/040/REPORT.md)
