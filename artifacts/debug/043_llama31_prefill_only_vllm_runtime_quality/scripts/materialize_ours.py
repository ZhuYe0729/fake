#!/usr/bin/env python3
"""Export one final Llama3.1 prefill policy as a phase-hetero checkpoint."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]; DEBUG=ROOT/"artifacts/debug/043_llama31_prefill_only_vllm_runtime_quality"
VLLM=Path("/home/agent/wja/project/my/cospaq/test/vllm")
CUTLASS=ROOT/"fake/kernels/cutlass/cutlass_wrapper"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--policy",required=True); a=p.parse_args()
    manifest=json.loads((DEBUG/"manifest/policies.json").read_text()); item=next(x for x in manifest["policies"] if x["label"]==a.policy)
    if item["kind"]!="ours": raise ValueError(a.policy)
    output=ROOT/item["checkpoint"]
    exported_policy = output / "phase_hetero_policy.json"
    if exported_policy.exists():
        if json.loads(exported_policy.read_text()) != json.loads((ROOT / item["policy_json"]).read_text()):
            raise RuntimeError(f"existing checkpoint policy differs: {output}")
        return
    exporter=VLLM/"artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
    output.parent.mkdir(parents=True,exist_ok=True)
    policy = json.loads((ROOT / item["policy_json"]).read_text())
    methods = [policy["default_prefill_method"], policy["default_decode_method"]]
    methods.extend(method for pair in policy["method_map"].values() for method in pair.values())
    command = [sys.executable,str(exporter),"--model-path",manifest["model_path"],"--policy-json",str(ROOT/item["policy_json"]),"--output-dir",str(output),"--cutlass-wrapper-path",str(CUTLASS)]
    if output.exists():
        command.append("--force")
    if any(method.startswith("sparse_") for method in methods):
        command.append("--prune")
    subprocess.run(command,check=True)
    if json.loads(exported_policy.read_text()) != json.loads((ROOT / item["policy_json"]).read_text()):
        raise RuntimeError(f"exported checkpoint policy differs: {output}")
if __name__=="__main__": main()
