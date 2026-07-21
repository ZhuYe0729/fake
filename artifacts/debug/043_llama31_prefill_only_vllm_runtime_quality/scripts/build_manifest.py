#!/usr/bin/env python3
"""Build the Llama3.1 real-vLLM prefill-only policy inventory."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEBUG = ROOT / "artifacts/debug/043_llama31_prefill_only_vllm_runtime_quality"
MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
SUMMARY = ROOT / "artifacts/exports/vllm/ours/llama3.1-8b-instruct/pareto_summary/summary.md"
UNIFORM = ROOT / "artifacts/exports/vllm/baselines/llama3.1-8b-instruct/checkpoints"
POLICIES = ROOT / "artifacts/debug/038_llama31_8b_instruct_prefill_only_pareto/pareto/policies"
CHECKPOINTS = DEBUG / "checkpoints"

def prefill_rows():
    lines = SUMMARY.read_text().splitlines(); idx = next(i for i, x in enumerate(lines) if x.startswith("| scenario |"))
    keys = [x.strip() for x in lines[idx].strip().strip("|").split("|")]; rows=[]
    for line in lines[idx+2:]:
        if not line.startswith("|"): break
        row = dict(zip(keys, [x.strip() for x in line.strip().strip("|").split("|")]))
        if row["scenario"].startswith("prefill-only"): rows.append(row)
    return rows

def rel(path: Path) -> str:
    try: return str(path.relative_to(ROOT))
    except ValueError: return str(path)

def main():
    uniform = {"dense_bf16": MODEL, "dense_nvfp4": UNIFORM / "uniform_dense_nvfp4", "marlin_nvfp4": UNIFORM / "uniform_marlin_nvfp4", "sparse_bf16": UNIFORM / "uniform_sparse_bf16", "sparse_nvfp4": UNIFORM / "uniform_sparse_nvfp4"}
    items=[]
    for row in prefill_rows():
        label=row["policy"]; item={"label":label,"family":row["family"],"recommended_use":row["recommended use"],"e2e_ms":float(row["E2E ms"]),"speedup":float(row["speedup"]),"speed_source":row["speed source"]}
        if label in uniform:
            if not uniform[label].exists(): raise FileNotFoundError(uniform[label])
            item.update(kind="uniform", checkpoint=rel(uniform[label]))
        else:
            point=int(label.rsplit("_",1)[1]); policy=POLICIES/f"point_{point:03d}.json"
            if not policy.exists(): raise FileNotFoundError(policy)
            item.update(kind="ours", checkpoint=rel(CHECKPOINTS/label), policy_json=rel(policy), policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
        items.append(item)
    if len(items)!=12: raise RuntimeError(f"expected 12 policies, got {len(items)}")
    out=DEBUG/"manifest/policies.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"model_path":str(MODEL),"policies":items},indent=2)+"\n")
    print(out)
if __name__ == "__main__": main()
