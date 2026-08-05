# Llama3.1 ablation consolidation plan

## Goal

Consolidate all completed Llama3.1-8B-Instruct ablations into one portable
debug artifact, including raw measurements, machine-readable summaries,
figures, tables, provenance, and a paper-ready `ablation.tex` containing only
the ablation section.

## Included experiments

1. Modeling ablation (064): unseen-shape kernel prediction, policy-level E2E
   composition, and quality prediction.
2. Quality calibration-size sensitivity (065): repeated nested subsets,
   labeling cost, frozen main configuration, and full 72-policy audit.
3. Runtime overlap ablation (066): formal sync-vs-async E2E latency and direct
   CUDA-event hidden-copy profile.

## Steps and verification

1. Create `artifacts/debug/067_llama31_ablation_consolidation` with stable
   `data/`, `figures/`, `tables/`, `docs/`, and `scripts/` subdirectories.
   - Verify: no checkpoint or other reproducible multi-GB file is copied.
2. Copy the authoritative raw and summary artifacts and generate a SHA256
   provenance manifest recording every source and destination.
   - Verify: all copied hashes match their sources.
3. Write a consolidated Markdown summary that defines each metric, protocol,
   result, and paper recommendation.
   - Verify: every reported number is reproduced from copied CSV/JSON data.
4. Write `ablation.tex`, following the terminology and section organization of
   `references/wu.tex`, while replacing all ablation placeholders with measured
   values.
   - Verify: it is a section fragment rather than a standalone document; all
     referenced tables/figures are present in the consolidation directory.
5. Run structural checks and, if a LaTeX engine is available, compile a small
   wrapper document without modifying the paper draft.

## Presentation choice

- Use one compact table for kernel/E2E/quality modeling ablations.
- Use the paper-facing calibration curve capped at 64 policies, while retaining
  and documenting the full 72-policy audit curve and data.
- Use a separate runtime table because latency and hidden-copy metrics are not
  prediction errors and should not be merged into the modeling table.
