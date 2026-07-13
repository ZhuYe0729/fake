#!/usr/bin/env python3
"""Compare raw predictor aggregation with already measured uniform vLLM baselines."""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path

METHOD = {"p00":"dense_bf16", "p01":"dense_nvfp4", "p02":"sparse_bf16", "p04":"marlin_nvfp4"}
def read(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scenario",choices=("prefill_only","prefill_decode"),required=True);p.add_argument("--output-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    predicted={r["policy_id"]:r for r in read(a.output_root/"speed_model"/f"{a.scenario}_predictions.csv")}
    baseline=a.output_root.parents[3]/"baselines/llama2-7b-chat/results/summary/speed_summary.csv"
    actual={r["method"]:r for r in read(baseline) if r["scenario"]==a.scenario}
    rows=[]
    for pid,method in METHOD.items():
        raw=float(predicted[pid]["raw_predicted_linear_ms"]); e2e=float(actual[method]["e2e_median_ms"])
        rows.append({"policy_id":pid,"uniform_method":method,"raw_predicted_linear_ms":raw,"measured_vllm_e2e_ms":e2e,"absolute_error_ms":abs(raw-e2e),"relative_error":abs(raw-e2e)/e2e})
    mae=sum(float(x["absolute_error_ms"]) for x in rows)/len(rows); mape=sum(float(x["relative_error"]) for x in rows)/len(rows)
    out=a.output_root/"speed_model"/f"{a.scenario}_uniform_validation.csv"
    with out.open("w",newline="") as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    print(f"{a.scenario}: raw-linear vs E2E MAE={mae:.3f} ms, MAPE={mape:.3%}; wrote {out}")
if __name__=="__main__":main()
