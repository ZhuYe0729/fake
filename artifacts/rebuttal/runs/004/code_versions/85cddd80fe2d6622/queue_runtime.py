"""One GPU, immutable command snapshots, reusable temporary exports."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import MODEL,RUN,WRAPPER,write_json

TEMP=Path('/dev/shm/daso_rebuttal_004_20261025')


def invoke(command,label):
    stamp=str(time.time_ns());log=RUN/f'logs/{label}_{stamp}.log'
    files=sorted((RUN/'code').glob('*.py'));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot=RUN/'code_versions'/version
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for p in files:shutil.copy2(p,snapshot/p.name)
    write_json(log.with_suffix('.command.json'),{'command':list(map(str,command)),'code_version':version,'code_hashes':hashes,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    start=time.perf_counter()
    with log.open('w') as f:result=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
    if result.returncode:raise RuntimeError(f'exit {result.returncode}: {log}')


def checkpoint(policy):
    if policy.startswith('uniform_'):
        old=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/policy
        if (old/'model.safetensors').exists():return old
    target=TEMP/'checkpoints'/policy
    if (target/'model.safetensors').exists():return target
    assert shutil.disk_usage('/dev/shm').free>32*1024**3
    command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',RUN/f'configs/policies/{policy}.json',
             '--output-dir',target,'--cutlass-wrapper-path',WRAPPER]
    content=(RUN/f'configs/policies/{policy}.json').read_text()
    for method in ['sparse_bf16','sparse_nvfp4']:
        if method in content:command+=['--canonical-'+method.replace('_','-')+'-state',Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt']
    invoke(command,policy+'_export')
    metadata=RUN/f'results/exports/{policy}';metadata.mkdir(parents=True,exist_ok=True)
    for p in target.glob('*.json'):shutil.copy2(p,metadata/p.name)
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',action='store_true');args=p.parse_args()
    assert (RUN.parent/'003/results/ready_to_seal.json').exists(),'finish run003 first'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    design=json.loads((RUN/'configs/design.json').read_text())
    state=RUN/('results/profile_worker.json' if args.profiles else 'results/runtime_worker.json')
    write_json(state,{'status':'running','gpu':6,'pid':os.getpid()})
    try:
        for policy in design['policies']:
            if args.profiles and policy not in ['transfer_daso8_prefill_decode_01','phase_sq_int8_w4a16_ours']:continue
            model=None
            for mode in (design['uniform_modes'] if policy.startswith('uniform_') else design['modes']):
                marker=RUN/f'results/complete/{policy}_{mode}_{"profile" if args.profiles else "timing"}.json'
                if marker.exists():continue
                if model is None:model=checkpoint(policy)
                command=[sys.executable,RUN/'code/measure_runtime.py','--checkpoint',model,'--policy',policy,'--mode',mode]
                if args.profiles:command+=['--profile']
                invoke(command,policy+'_'+mode+('_profile' if args.profiles else '_timing'))
                write_json(marker,{'complete':True});print(policy,mode,'complete',flush=True)
            # Preserve the three heterogeneous exports for mixed-token reuse;
            # only release after run006 has acquired its own reproducible copy.
        write_json(state,{'status':'complete','gpu':6,'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
