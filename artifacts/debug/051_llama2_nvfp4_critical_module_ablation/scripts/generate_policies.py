#!/usr/bin/env python3
"""Generate q120 leave-one-protected-module NVFP4 ablations."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; DEBUG=ROOT/'artifacts/debug/051_llama2_nvfp4_critical_module_ablation'; BASE=ROOT/'artifacts/debug/047_llama2_prefill_mechanism_quality_debug/policies/q120.json'
def main():
    base=json.loads(BASE.read_text()); kept=sorted(k for k,v in base['method_map'].items() if v['prefill_method']=='dense_bf16'); out=DEBUG/'policies';out.mkdir(parents=True,exist_ok=True);manifest=[]
    for i,name in enumerate(kept):
        p=json.loads(json.dumps(base));p['policy_id']=f'q120_plus_{i:02d}';p['policy_kind']='leave_one_protected_module';p['method_map'][name]['prefill_method']='dense_nvfp4';path=out/f'{p["policy_id"]}.json';path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');manifest.append({'policy_id':p['policy_id'],'changed_module':name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    p=json.loads(json.dumps(base)); p['policy_id']='q128_phase'; p['policy_kind']='phase_hetero_uniform_dense_nvfp4'
    for value in p['method_map'].values(): value['prefill_method']='dense_nvfp4'
    path=out/'q128_phase.json';path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');manifest.append({'policy_id':p['policy_id'],'changed_module':'all eight retained modules','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (DEBUG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
