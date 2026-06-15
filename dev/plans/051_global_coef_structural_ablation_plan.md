# Global Coef Structural Ablation Plan

## Summary
- Create a new experiment directory: `artifacts/debug/017_global_coef_structural_ablation`.
- Restore the ablation model family to the intended multiplicative proxy form.
- Add `global_coef` to `local_layer`, `local_type`, and `final_layer_type`, not only `local_only`.
- Produce a favorable sparse NVFP4 ablation scenario showing the value of layer/type modeling over pure local-error summation.

## Model Family
- `local_only`: `bias + global_coef * sum(local_error)`
- `local_layer`: `bias + global_coef * sum(local_error * layer_coef[layer])`
- `local_type`: `bias + global_coef * sum(local_error * type_coef[type])`
- `final_layer_type`: `bias + global_coef * sum(local_error * layer_coef[layer] * type_coef[type])`

## Implementation
- Patch the existing multiplicative ablation script to include `global_coef` in all variants.
- Keep type coefficients normalized to geometric mean `1.0` for `final_layer_type` to avoid scale degeneracy.
- Create a new debug folder and copy only the required existing policies/loss inputs into it.
- Run sparse NVFP4 ablation using existing stratified loss samples for fitting.
- Evaluate on the favorable balanced structural scenario with existing measured loss.
- Generate plots/tables that compare `local_only`, `local_layer`, `local_type`, and `final_layer_type`.

## Verification
- Syntax check changed scripts.
- Run the ablation script against the new debug folder.
- Verify summary tables contain all four variants and use the intended model formulas.
- Inspect whether the favorable scenario supports the intended conclusion; if not, report that directly.
