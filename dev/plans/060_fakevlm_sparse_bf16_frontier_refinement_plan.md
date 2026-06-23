# 060 FakeVLM Sparse BF16 Frontier Refinement Plan

## Summary
- Refine the batch-16 measured Pareto curve around the uniform sparse BF16 baseline.
- Add existing global-Pareto policies P19, P20, and P21 between the currently measured P18 and P22.
- Preserve existing corrected-NLL reports and emit independent refined report artifacts.

## Implementation
1. Extend validation-point selection with explicit additional point indices.
2. Select the original eight batch-16 points plus P19, P20, and P21.
3. Measure added-point E2E speed serially on one GPU, with no concurrent GPU work.
4. Measure full 5000-sample FakeClue accuracy for the three points in parallel on separate GPUs.
5. Rebuild the joined validation table and generate report files with the `refined_sparse_bf16` suffix.

## Verification
- Require exactly 11 selected batch-16 policies and new speed/accuracy rows for P19, P20, and P21.
- Require no failed/skipped linear replacements and 5000 evaluated samples per added policy.
- Confirm the refined curve contains measured points between P18 and P22.
- Keep all existing corrected-NLL report artifacts unchanged.

## Assumptions
- P19-P21 are the appropriate refinement points because they are existing policies on the corrected batch-16 predicted global Pareto frontier.
- The report should show measured outcomes honestly; sparse BF16 may remain nondominated if the fitted NLL objective does not preserve downstream accuracy ranking in this region.
