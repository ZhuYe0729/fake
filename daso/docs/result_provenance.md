# Paper result provenance

`results/paper/manifest.json` is the machine-readable inventory. Every archived
file has a SHA-256 digest, provenance level, and original experiment source.
Machine-local paths in text results are deterministically replaced by logical
`model://`, `backend://`, `legacy-artifact://`, or `legacy-host://` identifiers;
numeric measurements and policy contents are unchanged.

The archive contains:

- RTX 5090 results for Llama-2-7B-Chat and Llama-3.1-8B-Instruct;
- RTX PRO 6000 cross-platform results;
- quality, kernel, calibration-size, and runtime-overlap ablations;
- appendix random, neighborhood, and leverage-guided Pareto challengers;
- frozen calibration policies, local-error tables, NLL labels, fitted models,
  action-support tables, final policies, tables, and figures.

It excludes model/canonical checkpoints, datasets, extension caches, temporary
exports, failed runs, and historical debug variants. These are reproducible or
external inputs and would otherwise dominate repository size.

Known limitations are preserved rather than hidden: several Llama-2 generation
baselines retain aggregate measurements but not all per-sample outputs, and the
manually authored motivation/overview PDFs were not present under
`artifacts/paper`. The detailed historical audit is retained as
`results/paper/main_table_provenance_report.md`.

Runtime overlap is reported as `1 - 1.92 / 354.94 = 99.46%`, consistent with
the displayed table components.

Archive validation separately reports structural checks, source-derived paper
cell checks, and complete manifest hashes.  Numeric checks locate a concrete
policy row and metric before comparing the formatted value with both the
canonical table and final paper source; they are not global string searches.

