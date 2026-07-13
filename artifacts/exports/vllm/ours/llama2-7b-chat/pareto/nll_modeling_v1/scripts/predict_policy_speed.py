#!/usr/bin/env python3
"""Aggregate raw KernelLatencyPredictor latency for calibration policies (no E2E fit)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
sys.path[:0] = [str(ROOT), str(CUTLASS), str(CUTLASS / "modeling")]
from modeling.kernel_predictor import DEFAULT_MODEL_ROOT, KernelLatencyPredictor  # noqa: E402

KERNEL = {"w4a16_ours": "marlin_nvfp4"}
SHAPE = {"qkv_proj": (12288,4096), "o_proj": (4096,4096), "gate_up_proj": (22016,4096), "down_proj": (4096,11008)}
SCENARIO = {"prefill_only": (8*2048, 8, 0), "prefill_decode": (16*2048, 16, 80)}

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--scenario", choices=tuple(SCENARIO), required=True); p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1]); p.add_argument("--predictor-root", type=Path, default=DEFAULT_MODEL_ROOT); a=p.parse_args()
    prefill_m, decode_m, decode_steps = SCENARIO[a.scenario]; predictor=KernelLatencyPredictor(model_root=a.predictor_root, kernels=("dense_bf16","dense_nvfp4","sparse_bf16","sparse_nvfp4","marlin_nvfp4"))
    latency = {}
    for typ, (n, k) in SHAPE.items():
        latency["prefill", typ] = {x.kernel: float(x.latency_ms) for x in predictor.predict(prefill_m, n, k).candidates if x.supported and x.latency_ms is not None}
        if decode_steps:
            latency["decode", typ] = {x.kernel: float(x.latency_ms) for x in predictor.predict(decode_m, n, k).candidates if x.supported and x.latency_ms is not None}
    rows=[]
    for path in sorted((a.output_root/"policies"/a.scenario).glob("p*.json")):
        policy=json.loads(path.read_text()); total=0.
        for name, entry in policy["method_map"].items():
            typ=name.rsplit(".",1)[-1]; pre=latency["prefill", typ]
            dec=latency["decode", typ] if decode_steps else {}
            total += pre[KERNEL.get(entry["prefill_method"],entry["prefill_method"])]
            if decode_steps: total += decode_steps*dec[KERNEL.get(entry["decode_method"],entry["decode_method"])]
        rows.append({"policy_id":policy["policy_id"],"scenario":a.scenario,"raw_predicted_linear_ms":total,"e2e_validation_ms":"","note":"raw kernel aggregate; no fitted E2E correction"})
    out=a.output_root/"speed_model"; out.mkdir(parents=True,exist_ok=True)
    with (out/f"{a.scenario}_predictions.csv").open("w",newline="") as f: w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    print(f"wrote {len(rows)} raw predictions")
if __name__ == "__main__": main()
