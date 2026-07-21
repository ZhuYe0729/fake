#!/usr/bin/env python3
"""Solve discrete prefill-only policies using the validated speed and quality proxies."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from scenario import EXP, SOURCE_038, METHODS, KERNEL, TYPES

STEP = .002
REQUESTED_BUDGETS = (0., .005, .01, .02, .04, .06, .08, .12, .18, .25, .35, .5, .7, 1., 1.3, 1.7)

def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))

def pava(values: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([value, 1.])
        while len(blocks) > 1 and blocks[-2][0]/blocks[-2][1] > blocks[-1][0]/blocks[-1][1]:
            right = blocks.pop(); blocks[-1][0] += right[0]; blocks[-1][1] += right[1]
    return [total/count for total, count in blocks for _ in range(int(count))]

def corrected(train: list[tuple[float,float]], x: float) -> float:
    pairs = sorted(train); xs = [a for a,_ in pairs]; ys = pava([b for _,b in pairs])
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for index in range(1, len(xs)):
        if x <= xs[index]:
            ratio = (x-xs[index-1])/(xs[index]-xs[index-1]); return ys[index-1] + ratio*(ys[index]-ys[index-1])
    raise AssertionError

def main() -> None:
    model = json.loads((EXP / "reports/quality/model.json").read_text()); scale = float(model["feature_scale"])
    coefficient = lambda mi,b,t: max(0., float(model["global"]) + float(model["method"][mi]) + float(model["bucket"][b]) + float(model["type"][t])) / scale
    local = {(b,t,"dense_bf16"): 0. for b in range(4) for t in TYPES}
    source = EXP.parent.parent / "057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct/local_errors"
    for method in METHODS[1:]:
        for row in read(source / f"prefill_{method}.csv"):
            local[int(row["layer_bucket"]), row["fused_type"], method] = float(row["output_rel_mse"])
    action = read(SOURCE_038 / "action_support.csv")
    latency = {(r["module_name"],r["kernel"]):float(r["latency_ms"]) for r in action if r["supported"] == "True"}
    modules = []
    for layer in range(32):
        for group, typ in (("self_attn","qkv_proj"),("self_attn","o_proj"),("mlp","gate_up_proj"),("mlp","down_proj")):
            name = f"model.layers.{layer}.{group}.{typ}"; bucket = layer // 8; options = []
            for mi, method in enumerate(METHODS):
                quality = local[bucket,typ,method] * coefficient(mi,bucket,TYPES.index(typ))
                # A fitted ReLU coefficient can be exactly zero.  That does
                # not make a compressed action admissible at the exact BF16
                # endpoint: only dense BF16 occupies the zero-quality bin.
                qbin = 0 if method == "dense_bf16" else max(1, math.ceil(quality/STEP))
                options.append((qbin, quality, latency[name,KERNEL[method]], method))
            modules.append((name, options))
    max_bin = max(1, int(max(REQUESTED_BUDGETS)/STEP)); dp = [math.inf]*(max_bin+1); dp[0] = 0.; back = []
    for _, options in modules:
        nxt = [math.inf]*(max_bin+1); choice = [None]*(max_bin+1)
        for old, value in enumerate(dp):
            if not math.isfinite(value): continue
            for oi,(qbin,_,cost,_) in enumerate(options):
                new = min(max_bin, old+qbin); candidate = value+cost
                if candidate < nxt[new]: nxt[new] = candidate; choice[new] = (old,oi)
        dp = nxt; back.append(choice)
    calibration = read(EXP / "speed/calibration/calibration.csv")
    train = [(float(r["raw_predicted_linear_ms"]),float(r["e2e_median_ms"])) for r in calibration if r["split"] == "train"]
    policy_dir = EXP / "pareto/policies"; policy_dir.mkdir(parents=True, exist_ok=True); result=[]; seen=set()
    for budget in REQUESTED_BUDGETS:
        limit = min(max_bin, int(math.floor(budget/STEP))); state = min(range(limit+1), key=lambda value: dp[value]); picks=[]
        for index in range(len(modules)-1,-1,-1):
            item = back[index][state]
            if item is None: raise RuntimeError("unreachable DP state")
            state, option = item; picks.append(option)
        picks.reverse(); signature=tuple(picks)
        if signature in seen: continue
        seen.add(signature); policy_id=f"point_{len(result):03d}"
        mapping={name:{"prefill_method":opts[pick][3],"decode_method":opts[pick][3]} for (name,opts),pick in zip(modules,picks)}
        raw=sum(opts[pick][2] for (_,opts),pick in zip(modules,picks)); quality=sum(opts[pick][1] for (_,opts),pick in zip(modules,picks))+float(model["bias"])
        policy={"policy_id":policy_id,"scenario":"prefill_only","policy_kind":"predicted_pareto_screening","default_prefill_method":"dense_bf16","default_decode_method":"dense_bf16","modules_to_not_convert":["lm_head"],"method_map":mapping}
        (policy_dir/f"{policy_id}.json").write_text(json.dumps(policy,indent=2,sort_keys=True)+"\n")
        result.append({"point_index":len(result),"policy_id":policy_id,"quality_budget":budget,"predicted_delta_nll":max(0.,quality),"raw_predicted_linear_ms":raw,"corrected_e2e_prediction_ms":corrected(train,raw),**{f"count_{m}":sum(v["prefill_method"]==m for v in mapping.values()) for m in METHODS}})
    output=EXP/"pareto/predicted_points.csv"
    with output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    (EXP/"pareto/metadata.json").write_text(json.dumps({"quality_model":"058 pure-prefill canonical real-vLLM NLL ReLU proxy","speed_model":"038 Llama3 kernel predictor plus 058 canonical phase E2E calibration","status":"screening only; measured closure required"},indent=2)+"\n")
    print(output)
if __name__ == "__main__": main()
