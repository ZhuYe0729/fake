#!/usr/bin/env python3
"""Create quality-ranked BF16→dense-NVFP4 bridge policies near the uniform baseline."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from scenario import EXP, LOCAL_ERRORS, SOURCE_038, METHODS, TYPES, KERNEL

COUNTS = (72, 88, 104, 120)

def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))

def main() -> None:
    model = json.loads((EXP / "reports/quality/model.json").read_text()); scale = float(model["feature_scale"])
    coef = lambda bucket,typ: max(0., float(model["global"])+float(model["method"][1])+float(model["bucket"][bucket])+float(model["type"][TYPES.index(typ)]))/scale
    local = {(int(r["layer_bucket"]),r["fused_type"]):float(r["output_rel_mse"]) for r in read(LOCAL_ERRORS / "prefill_dense_nvfp4.csv")}
    latency = {(r["module_name"],r["kernel"]):float(r["latency_ms"]) for r in read(SOURCE_038 / "action_support.csv") if r["supported"]=="True"}
    candidates=[]
    for layer in range(32):
        for group,typ in (("self_attn","qkv_proj"),("self_attn","o_proj"),("mlp","gate_up_proj"),("mlp","down_proj")):
            name=f"model.layers.{layer}.{group}.{typ}"; bucket=layer//8; penalty=local[bucket,typ]*coef(bucket,typ); saving=latency[name,"dense_bf16"]-latency[name,"dense_nvfp4"]
            candidates.append((penalty/max(saving,1e-9),name,penalty,saving))
    candidates.sort()
    summary=[]; directory=EXP/"pareto/policies"; directory.mkdir(parents=True,exist_ok=True)
    for count in COUNTS:
        selected={name for _,name,_,_ in candidates[:count]}; label=f"bridge_dense_nvfp4_{count:03d}"
        mapping={name:{"prefill_method":"dense_nvfp4" if name in selected else "dense_bf16","decode_method":"dense_nvfp4" if name in selected else "dense_bf16"} for _,name,_,_ in candidates}
        policy={"policy_id":label,"scenario":"prefill_only","policy_kind":"dense_nvfp4_bridge","default_prefill_method":"dense_bf16","default_decode_method":"dense_bf16","modules_to_not_convert":["lm_head"],"method_map":mapping}
        (directory/f"{label}.json").write_text(json.dumps(policy,indent=2,sort_keys=True)+"\n")
        summary.append({"policy_id":label,"dense_nvfp4_modules":count,"predicted_delta_nll":sum(x[2] for x in candidates[:count]),"raw_predicted_linear_ms":sum(latency[name,"dense_nvfp4" if name in selected else "dense_bf16"] for _,name,_,_ in candidates)})
    output=EXP/"pareto/dense_nvfp4_bridge.csv"
    with output.open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(output)
if __name__=="__main__": main()
