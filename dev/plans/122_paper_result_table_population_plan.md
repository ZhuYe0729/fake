# Paper result table population plan

## Goal

Populate the RTX 5090 portion of `artifacts/debug/060_two_model_two_scenario_result_consolidation/result.tex` from the retained canonical measurements, using one max-speed and one balanced Ours point per model/scenario.

## Selection rules

- `Ours (Max speed)`: fastest retained measured Ours point in that scenario.
- `Ours (Balanced)`: the strongest retained quality-preserving Ours point that provides a compelling comparison with the uniform methods; selection may differ by prefill-only versus prefill-decode because table columns represent separate scenarios.
- Compute prefill-decode TTFT/TPOT/E2E speedups from each canonical scenario's common dense-BF16 uniform baseline file.
- Do not fill RTX PRO 6000 because no corresponding retained measurements are available.

## Selected points

- Llama2 prefill-only: max `point_024`, balanced `point_017`.
- Llama2 prefill-decode: max `b8o64009`, balanced `b8o64004`.
- Llama3 prefill-only: max `point_014`, balanced `bridge_dense_nvfp4_120`.
- Llama3 prefill-decode: max `point_011`, balanced `point_005`.

## Verification

1. Derive every inserted figure from a retained CSV and round consistently to two decimals.
2. Check every table row has exactly 13 columns and LaTex syntax remains structurally valid.
3. Record selections, sources, and caveats in the matching implementation log.
