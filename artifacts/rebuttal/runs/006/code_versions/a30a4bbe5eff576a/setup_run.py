"""Isolate the mixed-token runtime extension and freeze burst workloads."""
import hashlib
import json
import shutil
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def main():
    source=RUN.parent/'004'
    for d in ['configs/policies','data','reference','results','logs','code/compat']:(RUN/d).mkdir(parents=True,exist_ok=True)
    for name in ['common.py','eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py','resident.py']:
        shutil.copy2(source/'code'/name,RUN/'code'/name)
    shutil.copytree(source/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    for name in ['activation_scales.pt','profile_weights.pt']:(RUN/'data'/name).symlink_to(source/'data'/name)
    design=json.loads((source/'configs/design.json').read_text());policies=design['policies']
    for name in policies:shutil.copy2(source/f'configs/policies/{name}.json',RUN/f'configs/policies/{name}.json')
    frozen={'policies':policies,'warmups':2,'iterations':5,'kv_bytes':6*1024**3,
        'cases':[{'id':'chunk1024_b8_s2048_o64','prompts':[2048]*8,'output':64,'token_budget':1024},
                 {'id':'chunk4096_b16_heterogeneous_o64','prompts':[512,4096]*8,'output':64,'token_budget':4096},
                 {'id':'chunk1024_b8_heterogeneous_o256','prompts':[512,2048]*4,'output':256,'token_budget':1024}],
        'engine':{'max_model_len':4360,'max_num_seqs':16},'speed_seed':144907,
        'scope':'Actual vLLM V1 chunked-prefill scheduler; closed-loop offline burst, all arrivals at t0. Real mixed forwards required; no HTTP/network or open-loop arrival claim.',
        'runtime':'experiment-local dual-resident representation, rowwise scheduled-token phase routing; original global-phase streaming runtime not claimed mixed-compatible',
        'quality':'Separate validation-text teacher forcing versus homogeneous, nonchunked execution; numerical routing audits and uniform no-op control first'}
    (RUN/'configs/design.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (RUN/'reference/input_hashes.json').write_text(json.dumps({str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (RUN/'configs').rglob('*.json')},indent=2)+'\n')
    (RUN/'README.md').write_text('# Run006 — 真实混合 token 调度实验\n\n独立dual-resident变体；实际vLLM chunked prefill。原实现限制和新增内存/调度开销分开报告。\n')


if __name__=='__main__':main()
