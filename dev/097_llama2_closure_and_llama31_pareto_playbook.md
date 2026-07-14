# Llama2 Pareto closure and Llama-3.1-8B-Instruct execution playbook

## 1. Executive conclusion

The method is now established as a **surrogate-guided, measured-closure
workflow**:

1. build a model- and scenario-specific quality surrogate from cheap,
   controlled WikiText teacher-forced measurements;
2. predict kernel cost with a profile-calibrated roofline model and calibrate
   the aggregate prediction to real vLLM E2E latency;
3. solve a layer-/phase-heterogeneous discrete quality-constrained problem;
4. report only the Pareto frontier recomputed from measured speed and measured
   NLL or task quality, together with uniform methods.

It works for the two Llama2 serving scenarios in the sense that it produces
valid heterogeneous policies, predicts enough of their ranking to guide a
small measurement set, and yields measured mixed-policy trade-offs. It is **not
a theorem that heterogeneous compression strictly beats every uniform method**.
At a given workload, a uniform kernel may already be near the hardware optimum;
then the valid result is a finer trade-off or a tie, not a fabricated win.

The Llama2 prefill-only ARC result is the clearest example. At b=8, input=2048,
mixed point 8 reaches 1.213x dense-BF16 speed with ARC-Challenge `acc_norm`
0.4334, matched to dense BF16 within the full-evaluation uncertainty. Points
11--13 form a measured high-quality middle region: 1.635x/0.4403,
1.726x/0.4420, and 1.811x/0.4343. Uniform dense-NVFP4 remains a legitimate
1.867x/0.4283 endpoint. The final curve therefore contains both ours and
uniform points. Sparse-NVFP4 remains the unrestricted max-speed uniform
endpoint; this is a result of the available kernels/actions, not a solver bug.

## 2. What is fixed, and what is model/scenario-specific

| Component | Reuse as method | Refit/remeasure on Llama3.1 |
|---|---|---|
| Fused action abstraction | Yes: `qkv`, `o`, `gate_up`, `down` | Yes: derive dimensions/count from config |
| Runtime method mapping | Yes: logical `w4a16_ours` maps to Marlin runtime | Validate supported pairs and conversion kernels |
| Kernel surrogate form | Yes: profile-calibrated roofline + shape residual | Query new shapes; add/validate profiles when outside training support |
| Policy E2E correction | Yes: monotone policy-level correction | Fit separately for each scenario/model/runner protocol |
| WikiText quality target | Yes: fixed blocks and teacher forcing | Recompute local sensitivities and policy NLL labels |
| Quality proxy form | Yes: normalized local-error aggregation with method/bucket/type factors | Refit; do not copy Llama2 coefficients |
| DP/knapsack solver | Yes | Regenerate candidate actions, legal pairs, and quality budgets |
| Final plotting rule | Yes | Use new measured axes; union with new uniform baselines |

The important transfer boundary is: **the workflow transfers, fitted models and
policies do not**.

## 3. Llama2 closure: evidence and approved claims

### 3.1 Prefill-only (`b=8`, `input=2048`)

Final main result directory:
`artifacts/debug/037_llama2_prefill_only_pareto/`.

- Speed surrogate: raw kernel-sum prediction plus monotone E2E calibration.
  Independent held-out E2E MAE improved from 62.56 ms (raw dense-scaled) to
  11.31 ms; all-point LOO MAE is 10.78 ms. See
  `e2e_calibration_metrics.json`.
- Quality surrogate: 72 controlled calibration policies, fixed 54/18 train/
  holdout split; prefill-only holdout MAE 0.126 and Spearman 0.774. See
  `artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/reports/prefill_only/metrics.json`.
- Final speed/NLL axes: measured five-run vLLM median and fixed 100-block
  WikiText NLL; see `report/actual_nll_speed_summary.csv` and
  `report/pareto_speedup_vs_actual_wikitext_nll.png`.
- Final downstream prefill task: ARC-Challenge, 0-shot answer likelihood,
  complete 1,172-example set. See
  `arc_challenge/report/arc_challenge_speed_summary.csv`,
  `pareto_speedup_vs_arc_challenge.png`, and the high-quality plateau plot.

Approved wording: “At the evaluated prefill workload, mixed policies provide
a measured, fine-grained quality--speed frontier and match dense quality at
1.213x. The final frontier is computed jointly with uniform references.”

Avoid: “Heterogeneity always beats uniform compression,” or “point 8 strictly
outperforms dense BF16.” ARC standard errors are material at this dataset size.

### 3.2 Prefill-decode (`b=16`, `input=2048`, `output=80`)

The validated experiment trail is under
`artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/` and
`artifacts/debug/036_llama2_prefill_decode_intermediate_points/`.

- The objective pools phase error as
  `DeltaNLL_prefill + 80 * DeltaNLL_decode`; decode cannot be ignored for a
  generating scenario.
- Phase-heterogeneous legal actions and continuous vLLM phase switching are
  used during runtime tests.
- Real NLL/speed curves and PMPD task-quality summaries are retained under the
  paths above. Any paper table must select values only from the final,
  protocol-matched summary/plot, not an older runner variant.

Approved wording: “The same framework supports phase-dependent assignments;
its quality constraint includes both prefill and repeated decode error.”

### 3.3 Diagnosed limitation, not an unresolved implementation bug

The dense-NVFP4 bridge sweep showed that, with the current sparse artifacts and
kernels, going faster than uniform dense-NVFP4 requires many sparse actions.
The actual NLL rises substantially (for example, a 1.90x bridge policy had
DeltaNLL +0.247 versus dense-NVFP4 +0.026). Sparse-BF16 itself was slower than
dense-NVFP4 in this prefill workload. Improving that region requires better
sparse reconstruction/masks or a faster sparse kernel, not only a new DP
search heuristic.

## 4. Llama3.1 prerequisites already available

Model: `/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`.

Existing assets are useful starting points, but they are not Pareto evidence:

- Uniform prepared checkpoints, speed, PMPD data and summaries:
  `artifacts/exports/vllm/baselines/llama3.1-8b-instruct/`.
- Existing unconstrained predictor-only max-speed experiment:
  `artifacts/exports/vllm/ours/llama3.1-8b-instruct/`.
- The latter already demonstrates why baselines must be part of the final
  Pareto set: its prefill-only max-speed policy is 1.764x dense but only
  0.878x of the best uniform speed in its recorded summary.

Architecture must be read from `AutoConfig`, not copied from Llama2. The
existing policy generator correctly derives GQA QKV output as
`hidden + 2 * num_key_value_heads * head_dim`; for Llama3.1 it is 6144x4096,
not Llama2's 12288x4096. It also derives MLP dimensions and module count from
the config. Reuse its `linears()` logic.

## 5. Step-by-step Llama3.1 execution protocol

Create a new debug root, for example
`artifacts/debug/038_llama31_8b_instruct_pareto/`. Do not modify 033--037 or
mix its measurements into the new root.

### Gate A — freeze protocol and audit architecture

Inputs:

- baseline summaries and runners above;
- `AutoConfig` for the Llama3.1 model;
- current vLLM phase-heterogeneous extension.

Actions:

1. Freeze the two workloads: prefill-only `(b=8, in=2048, out=1 API token)`
   and prefill-decode `(b=16, in=2048, out=80)`.
2. Emit a machine-readable architecture manifest: layers, hidden/intermediate,
   attention heads, KV heads, head dimension, all 4 fused shapes per layer,
   and expected module count.
3. Query every `(shape, method)` and phase pair in `KernelLatencyPredictor`.
   Record support, conversion support, nearest-profile distance and predicted
   latency. Unsupported actions are removed; never silently treated as dense.
4. Freeze one runner setting per scenario, including vLLM version, memory
   utilization, eager/graph mode, prefix-cache setting, batch/input/output and
   fresh-process repeat count.

Pass condition: dense and every uniform method can be exported and smoke-loaded
by that exact runner; all future comparisons use the same setting.

### Gate B — establish uniform references first

Inputs: baseline root and Gate-A runner.

Actions:

1. Reproduce or rerun five uniform references: dense BF16, dense NVFP4,
   sparse BF16, sparse NVFP4, and Marlin/W4A16.
2. Use five fresh-process speed repetitions and record median plus all samples.
3. Record fixed WikiText NLL with the final tokenizer/model and at least one
   prompt-likelihood task score (ARC-Challenge for prefill-only).

Pass condition: a single CSV contains uniform speed, NLL, runner metadata,
and task quality. These are optimizer candidates and plot points from the
beginning, not post-hoc comparators.

### Gate C — build controlled quality calibration data

Inputs: Llama3.1 prepared compressed weights, fixed WikiText blocks.

Actions:

1. Recompute per-layer/fused-module local relative MSE for every logical
   method. For fused QKV and gate-up, average their constituent linear errors.
2. Generate approximately 72 controlled policies covering method, module type,
   layer bucket, sparse placement, and for prefill-decode phase combinations.
   Freeze a 54/18 train/holdout split before measuring labels.
3. Measure teacher-forced WikiText labels. Use prefill DeltaNLL for
   prefill-only; use `prefill + 80 * decode` for prefill-decode.
4. Fit the normalized pooled positive local+global model. Report holdout MAE,
   RMSE, Spearman and a calibration-vs-prediction scatter.

Pass condition: holdout rank correlation is useful for screening (target at
least the Llama2-level evidence, rather than an unvalidated fit). If weak,
add controlled calibration coverage; do not fit downstream PMPD scores.

### Gate D — validate and calibrate speed surrogate

Inputs: Gate-A action table, kernel predictor, 10--12 policy assignments that
span dense to sparse (not only candidate frontier points).

Actions:

1. Sum supported kernel and conversion latency per policy as the raw feature.
2. Measure fixed-protocol E2E latency for a training calibration subset and an
   independently held-out subset.
3. Fit a monotone policy-level raw-latency-to-E2E mapping. Evaluate both raw
   dense-scaled and corrected MAE on the held-out policies.
4. Keep raw policy ordering only if the correction is monotone; then it is
   safe to use the raw additive objective in DP and correct the displayed
   E2E axis. Otherwise solve using a validated alternative feature/model.

Pass condition: correction improves held-out error and no mixed/uniform
protocol mismatch is hidden in the comparison.

### Gate E — solve, then audit the endpoints

Actions:

1. Solve multiple quality budgets with the multiple-choice discretized DP:
   minimize predicted raw latency subject to predicted quality cost.
2. Emit policy JSON, per-module choices, action counts, predicted quality and
   raw/corrected speed for each point.
3. Explicitly evaluate three endpoint classes:
   - dense quality endpoint;
   - unconstrained per-module mixed minimum;
   - every uniform policy.
4. Refine only intervals where the corrected speed curve has a large gap or a
   task-quality transition. Use 3--5 bridge policies, not a blind exhaustive
   task sweep.

Pass condition: the report distinguishes “fastest mixed candidate” from
“global max speed after including uniform methods.”

### Gate F — measured closure and paper plots

Actions:

1. For every displayed mixed point, measure five-run E2E median and fixed
   WikiText NLL. Recompute a non-dominated set over mixed plus uniform rows.
2. For prefill-only, run full ARC-Challenge only for measured NLL frontier
   points plus intermediate points needed to make the task curve readable.
3. For prefill-decode, evaluate representative/non-dominated policies on the
   established CNN/DM, DialogSum and IWSLT PMPD protocol. Reuse continuous
   phase-switch runners; do not restart/reload per request.
4. Produce: full measured plot, high-quality zoom, CSV with all points,
   frontier-only table, and a runner/protocol footnote.

Pass condition: no final axis is a prediction; dominated samples may be shown
as light points or appendix rows, but the line is the measured union frontier.

## 6. Files to reuse and what each is for

| Need | Canonical reference |
|---|---|
| Detailed method/formulas | `dev/094_llama2_prefill_decode_pareto_design.md` |
| Quality feature/measurement implementation | `artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/scripts/evaluate_wikitext_nll.py` |
| Discrete solver structure | `artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/scripts/solve_predicted_pareto.py` |
| E2E calibration protocol | `artifacts/debug/037_llama2_prefill_only_pareto/scripts/run_calibration_point.sh`, `fit_e2e_calibrator.py` |
| Measured NLL frontier/report | `artifacts/debug/037_llama2_prefill_only_pareto/scripts/build_actual_nll_frontier.py` |
| ARC evaluator/report | `artifacts/debug/037_llama2_prefill_only_pareto/scripts/evaluate_arc_challenge.py`, `build_arc_challenge_report.py` |
| Llama3 architecture-aware policy generation | `artifacts/exports/vllm/ours/llama3.1-8b-instruct/scripts/generate_max_speed_policy.py` |
| Llama3 uniform baselines | `artifacts/exports/vllm/baselines/llama3.1-8b-instruct/README.md` |
| Llama3 PMPD runners | `artifacts/exports/vllm/ours/llama3.1-8b-instruct/scripts/run_full_isolated_pmpd.py` and baseline quality scripts |

Copy scripts into the new debug root before model-specific edits. Change only
paths, architecture manifest, legal action support, and frozen protocol; do
not edit the Llama2 evidence directories.

## 7. Failure prevention checklist

- **Do not use GPU 0** when another workload owns it; select only confirmed
  available GPUs and record physical IDs. Do not rely on `CUDA_VISIBLE_DEVICES`
  remapping alone in a wrapper chain.
- **Do not mix vLLM environments**: use the compression environment for
  preparation/export and the vLLM environment for serving benchmarks.
- **Do not mix `.8`, `.85`, `.9` memory utilization results** in one curve.
  If `.9` fails due KV reservation, choose and document one feasible value for
  the complete comparison set, then rerun affected anchors.
- **Use fresh-process repetitions for the speed number**; model load/compiler
  time and one request latency must be separated in logs.
- **Use direct policy-weight installation only for teacher-forced quality**.
  vLLM checkpoint exports are required for runtime speed and PMPD generation.
- **Treat a missing JSON/CSV as failed, not zero**. Launchers must validate
  completed files, JSON content/line counts and resume only valid shards.
- **Never infer a final Pareto point from interpolated speed or predicted NLL**.
  Interpolation is for selecting measurement candidates only.
- **Do not claim a strict task-score win below uncertainty**. Include the
  evaluation sample count and stderr where available.

## 8. Expected outcome and stopping rules

The desired outcome is a measured union Pareto curve. A successful transfer can
show any of the following:

- a mixed policy strictly dominates a uniform point;
- mixed policies fill useful speed/quality gaps while uniform methods remain
  endpoints;
- a uniform method is globally fastest, while mixed policies own a high-quality
  or intermediate region.

All three are valid. Stop refining a region once measured NLL/task quality and
speed show it is uniformly dominated. Escalate to sparse weight reconstruction
or sparse-kernel improvement only when the desired region is impossible with
the current action set; do not spend GPU budget repeatedly re-solving the same
discrete alternatives.
