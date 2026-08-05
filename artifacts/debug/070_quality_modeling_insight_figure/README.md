# Quality Modeling Insight Figure

This artifact builds the paper's Figure 3 from the frozen real-kernel
single-Linear measurements in `015_llama2_prefill_kernel_loss_modeling`.

## Scope

- Model: Llama-2-7B.
- Method: sparse NVFP4 through the real CUTLASS runtime kernel.
- Data: 128 WikiText-2 train calibration sequences, length 512.
- Local metric: module-output relative MSE.
- Global metric: measured full-model NLL increase when only one Linear is
  compressed.
- Rows: 32 layers x 7 projection types = 224 single-Linear interventions.

The figure is a diagnostic design observation. It is not presented as a
Llama-3.1 result or as a fused-module intervention.

## Outputs

- `quality_modeling_insight.pdf/png`: paper figure.
- `sparse_nvfp4_single_linear_data.csv`: filtered source rows.
- `layer_profile_summary.csv`: per-layer means and median-normalized profile
  values used by panel (a).
- `metrics.json`: source provenance and single-Linear/layer-profile
  correlations.

The paper figure uses an equal-width double-column, side-by-side layout, with
both panels sharing the full layer-0--31 horizontal range. Panel (a) follows
the FakeVLM-style layer-profile presentation: both layer means are normalized
by their respective medians so that their depth trends can be compared on one
axis. Panel (b) retains the measured value for every layer and projection type.
The source typography and strokes are moderately enlarged so that the
side-by-side panels remain readable if the figure is scaled to one column.

The same PDF/PNG is written to:

- `references/Figures/draft2/quality_modeling_insight.{pdf,png}`;
- `references/Figures/draft/quality.{pdf,png}`, matching the existing
  `wu.tex` include path.

`references/wu.tex` is read-only in the current workspace, so its existing
caption still mentions inference phase even though this figure only visualizes
local error, depth, and projection type. Use `paper_snippet.tex` when that file
becomes writable.

## TODO for a later paper refresh

- Repeat the single-module diagnostic on Llama-3.1-8B-Instruct or
  Llama-2-7B-Chat under the paper calibration protocol.
- Measure the current four fused runtime groups (`qkv`, `o`, `gate-up`,
  `down`) directly instead of aggregating the seven unfused projections.
- Keep the present figure unless the refreshed measurements materially change
  the qualitative conclusion.

## Reproduction

```bash
conda run -n wja-cospaq python \
  artifacts/debug/070_quality_modeling_insight_figure/plot_quality_modeling_insight.py
```
