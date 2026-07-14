# Llama-3.1 paper result bundle plan

## Objective

Package the completed Llama-3.1-8B-Instruct experiments into one table and
dataset-level Pareto figures suitable for result selection and paper drafting.

## Steps

1. Consolidate every measured prefill-only ARC point and every measured
   prefill-decode task point, including dense BF16 and all frozen uniform
   compression baselines.
2. Mark recommended mixed-policy candidates without hiding the remaining
   measured points; retain speed source/protocol provenance for each row.
3. Create one Markdown/CSV result table plus ARC, CNN/DM, DialogSum, and
   IWSLT Pareto figures.  Use comparable continuous closure speeds where
   available, and visibly distinguish legacy-only uniform speed coordinates.
4. Validate row coverage, graph assets, and numerical agreement with the
   source summaries.

## Success criteria

The export bundle has a self-contained `summary.md`, machine-readable CSV,
and four readable figures.  A reader can identify the recommended points and
can tell which speed results share the continuous closure protocol.
