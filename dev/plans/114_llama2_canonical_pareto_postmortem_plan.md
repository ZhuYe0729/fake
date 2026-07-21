# 114 Llama2 canonical Pareto postmortem plan

## Goal

Record the Llama2-7B-chat prefill-only Pareto recalibration journey, its root
causes, confirmed fixes, and a repeatable checklist for future models.

## Steps

1. Capture the final experimental contract and verified outcomes.
2. Distinguish invalid historical measurements from valid canonical results.
3. Document prevention checks for pruning, runtime, speed, quality modelling,
   extension builds, and downstream evaluation.
4. Link concrete scripts and immutable result artifacts for reproduction.

## Success criteria

The document lets a future experimenter decide whether a result is comparable
before spending GPU time, and provides recovery steps for every issue observed
in this run.
