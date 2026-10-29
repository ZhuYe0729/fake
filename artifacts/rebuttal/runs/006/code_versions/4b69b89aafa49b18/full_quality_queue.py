"""Check measured serving winners and baselines on the actual B/S/O cases."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import MODEL,RUN,WRAPPER,write_json
TEMP=Path('/dev/shm/daso_rebuttal_006_full_quality_20261025')


def invoke(command,label):
    log=RUN/f'logs/{label}_{time.time_ns()}.log';files=sorted((RUN/'code').glob('*.py'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot=RUN/'code_versions'/version;snapshot.mkdir(parents=True,exist_ok=True)
    for p in files:
        if not (snapshot/p.name).exists():shutil.copy2(p,snapshot/p.name)
    write_json(log.with_suffix('.command.json'),{'command':list(map(str,command)),'code_version':version,'code_hashes':hashes,'gpu':6})
    start=time.perf_counter()
    with log.open('w') as f:result=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
    if result.returncode:raise RuntimeError(f'exit {result.returncode}: {log}')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    import torch
    import numpy as np
    design=json.loads((RUN/'configs/design.json').read_text());data=RUN/'data/validation_16_4352.pt'
    if not data.exists():
        tokens=torch.load(RUN.parent/'001/data/wikitext_validation_tokens.pt',weights_only=True)
        starts=np.arange(len(tokens)//4352)*4352;np.random.default_rng(144953).shuffle(starts);starts=starts[:16].tolist()
        torch.save(torch.stack([tokens[s:s+4352] for s in starts]),data)
        write_json(RUN/'configs/full_quality_design.json',{'policies':design['policies'],'cases':design['cases'],
            'split':'WikiText validation','seed':144953,'starts':starts,'blocks':16,'length':4352,
            'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'nonoverlap_within_split':True,
            'reason':'Prior short numerical probe showed chunk-size dependent FP4 drift; directly evaluate all six benchmark policies on actual B/S/O before asserting quality of serving speedups.',
            'selection':'No policy changes or selection using these labels; all baseline/DASO/phase-control cases retained.'})
    state=RUN/'results/full_quality_worker.json';write_json(state,{'status':'running','gpu':6,'pid':os.getpid()})
    try:
        for policy in design['policies']:
            if all((RUN/f'results/full_quality/{policy}/{c["id"]}.json').exists() for c in design['cases']):continue
            model=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/policy
            owned=policy!='uniform_dense_bf16'
            if owned:
                model=TEMP/'checkpoints'/policy
                if not (model/'model.safetensors').exists():
                    while shutil.disk_usage('/dev/shm').free<24*1024**3:time.sleep(10)
                    command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',RUN/f'configs/policies/{policy}.json',
                        '--output-dir',model,'--cutlass-wrapper-path',WRAPPER]
                    content=(RUN/f'configs/policies/{policy}.json').read_text()
                    for method in ['sparse_bf16','sparse_nvfp4']:
                        if method in content:command+=['--canonical-'+method.replace('_','-')+'-state',Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt']
                    invoke(command,policy+'_full_quality_export')
                directory=RUN/f'results/exports_full_quality/{policy}';directory.mkdir(parents=True,exist_ok=True)
                for p in model.glob('*.json'):shutil.copy2(p,directory/p.name)
            for budget in [1024,4096]:
                if all((RUN/f'results/full_quality/{policy}/{c["id"]}.json').exists() for c in design['cases'] if c['token_budget']==budget):continue
                invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',policy,
                    '--budget',budget,'--full-quality'],policy+'_full_quality_'+str(budget))
            if owned:
                binary=model/'model.safetensors';size=binary.stat().st_size;binary.unlink()
                write_json(RUN/f'results/exports_full_quality/{policy}/binary_cleanup.json',{'path':str(binary),'bytes':size,
                    'reason':'all actual-workload quality cases complete; all source/log/CSV/result/document/metadata retained'})
            print(policy,'actual-workload quality complete',flush=True)
        write_json(state,{'status':'complete','gpu':6,'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
