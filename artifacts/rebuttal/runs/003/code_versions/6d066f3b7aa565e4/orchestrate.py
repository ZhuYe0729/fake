"""Pipeline same-card speed and three independent quality workers."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import MODEL, RUN, WRAPPER, write_json

TEMP=Path('/dev/shm/daso_rebuttal_003_20261025')


def invoke(command,label):
    stamp=str(time.time_ns());log=RUN/f'logs/{label}_{stamp}.log'
    files=sorted((RUN/'code').glob('*.py'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot=RUN/'code_versions'/version
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for p in files:shutil.copy2(p,snapshot/p.name)
    write_json(log.with_suffix('.command.json'),{'command':list(map(str,command)),
        'code_version':version,'code_hashes':hashes,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    started=time.perf_counter()
    with log.open('w') as handle:result=subprocess.run(list(map(str,command)),stdout=handle,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-started})
    if result.returncode:raise RuntimeError(f'exit {result.returncode}: {log}')


def checkpoint(policy):
    if policy.startswith('uniform_'):
        old=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/policy
        if (old/'model.safetensors').exists():return old
    target=TEMP/'checkpoints'/policy
    if (target/'model.safetensors').exists() and (target/'phase_hetero_export_provenance.json').exists():return target
    while shutil.disk_usage('/dev/shm').free<32*1024**3:
        print('waiting for quality workers to release exported weights',policy,flush=True);time.sleep(5)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        target.rename(target.with_name(target.name+'_previous_'+str(time.time_ns())))
    command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',
             RUN/f'configs/policies/{policy}.json','--output-dir',target,'--cutlass-wrapper-path',WRAPPER]
    content=(RUN/f'configs/policies/{policy}.json').read_text()
    for method in ['sparse_bf16','sparse_nvfp4']:
        if method in content:
            command.extend(['--canonical-'+method.replace('_','-')+'-state',
                Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt'])
    invoke(command,policy+'_export')
    evidence=RUN/f'results/exports/{policy}';evidence.mkdir(parents=True,exist_ok=True)
    for source in target.glob('*.json'):shutil.copy2(source,evidence/source.name)
    return target


def release(policy):
    target=TEMP/f'checkpoints/{policy}/model.safetensors'
    if target.exists():
        size=target.stat().st_size;target.unlink()
        write_json(RUN/f'results/exports/{policy}/temporary_binary_cleanup.json',{
            'bytes':size,'path':str(target),'reason':'speed and all preassigned quality checks completed',
            'rebuild':'archived policy + code/export_model.py + original model + canonical sparse states + activation scales',
            'preserved':'all code, logs, JSON/CSV, documents and export metadata'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['speed','quality0','quality1','quality2'],required=True)
    args=parser.parse_args();name=args.role
    assert os.environ['CUDA_VISIBLE_DEVICES']=={'speed':'3','quality0':'0','quality1':'4','quality2':'7'}[name]
    design=json.loads((RUN/'configs/evaluation_points.json').read_text());assert design['complete_design']
    state=RUN/f'results/worker_{name}.json'
    write_json(state,{'status':'running','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES']})
    try:
        if name=='speed':
            for policy,job in design['policies'].items():
                marker=RUN/f'results/speed_complete/{policy}.json'
                if marker.exists():continue
                print('speed started',policy,len(job['workloads']),flush=True)
                model=checkpoint(policy)
                invoke([sys.executable,RUN/'code/measure_matrix.py','--checkpoint',model,
                    '--policy',policy,'--mode','speed','--workloads',','.join(job['workloads'])],policy+'_speed')
                write_json(marker,{'complete':True,'policy':policy,'checkpoint':str(model)})
                print('speed complete',policy,flush=True)
            write_json(RUN/'results/all_speed_complete.json',{'complete':True})
        else:
            while True:
                unfinished=0;claimed=False
                for policy,job in design['policies'].items():
                    complete=RUN/f'results/quality_complete/{policy}.json'
                    if complete.exists():continue
                    unfinished+=1;ready=RUN/f'results/speed_complete/{policy}.json'
                    if not ready.exists():continue
                    lock=RUN/f'results/locks/{policy}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
                    with lock.open('a') as handle:
                        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:continue
                        if complete.exists():continue
                        print('quality started',policy,len(job['workloads']),flush=True)
                        model=Path(json.loads(ready.read_text())['checkpoint'])
                        assert (model/'model.safetensors').exists(),model
                        invoke([sys.executable,RUN/'code/measure_matrix.py','--checkpoint',model,
                            '--policy',policy,'--mode','quality','--workloads',','.join(job['workloads'])],policy+'_quality')
                        write_json(complete,{'complete':True,'policy':policy,'worker':name})
                        release(policy);claimed=True
                        print('quality complete',policy,flush=True)
                        break
                if not unfinished:break
                if not claimed:time.sleep(5)
        write_json(state,{'status':'complete','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES']})
    except Exception as error:
        write_json(state,{'status':'failed','pid':os.getpid(),'error':repr(error)})
        raise


if __name__=='__main__':main()
