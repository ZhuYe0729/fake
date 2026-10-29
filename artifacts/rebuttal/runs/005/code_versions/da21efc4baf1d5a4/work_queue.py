"""Pipeline one common speed GPU and independent validation workers."""
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

from common import MODEL,RUN,WRAPPER,write_json

TEMP=Path('/dev/shm/daso_rebuttal_005_20261025')


def invoke(command,label):
    stamp=str(time.time_ns());log=RUN/f'logs/{label}_{stamp}.log';files=sorted((RUN/'code').glob('*.py'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16];snapshot=RUN/'code_versions'/version
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for p in files:shutil.copy2(p,snapshot/p.name)
    write_json(log.with_suffix('.command.json'),{'command':list(map(str,command)),'code_version':version,'code_hashes':hashes,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    start=time.perf_counter()
    with log.open('w') as f:result=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
    if result.returncode:raise RuntimeError(f'exit {result.returncode}: {log}')


def cleanup():
    for path in (TEMP/'checkpoints').glob('*/model.safetensors'):
        pid=path.parent.name
        final=(RUN/'configs/selected_policies.json').exists()
        if not (RUN/f'results/{"test" if final else "validation"}_complete/{pid}.json').exists():continue
        if not (RUN/f'results/{"test_speed" if final else "calibration_speed"}_complete/{pid}.json').exists():continue
        try:size=path.stat().st_size;path.unlink()
        except FileNotFoundError:continue
        write_json(RUN/f'results/exports/{pid}/binary_cleanup_{time.time_ns()}.json',{'bytes':size,'path':str(path),
            'reason':'calibration speed and all validation finished; reconstruct if selected for independent final test',
            'preserved':'all code/logs/results/CSV/documents and metadata'})


def checkpoint(pid):
    if pid.startswith('uniform_'):
        old=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/pid
        if (old/'model.safetensors').exists():return old
    target=TEMP/'checkpoints'/pid
    if (target/'model.safetensors').exists():return target
    while shutil.disk_usage('/dev/shm').free<32*1024**3:cleanup();time.sleep(3)
    if target.exists():target.rename(target.with_name(pid+'_previous_'+str(time.time_ns())))
    command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',RUN/f'configs/policies/{pid}.json',
        '--output-dir',target,'--cutlass-wrapper-path',WRAPPER]
    content=(RUN/f'configs/policies/{pid}.json').read_text()
    for method in ['sparse_bf16','sparse_nvfp4']:
        if method in content:command+=['--canonical-'+method.replace('_','-')+'-state',Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt']
    invoke(command,pid+'_export');metadata=RUN/f'results/exports/{pid}';metadata.mkdir(parents=True,exist_ok=True)
    for p in target.glob('*.json'):shutil.copy2(p,metadata/(str(time.time_ns())+'_'+p.name))
    return target


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['speed','validation','test_speed','test'],required=True);args=p.parse_args()
    assert (RUN.parent/'003/results/ready_to_seal.json').exists()
    role=args.role;gpu=os.environ['CUDA_VISIBLE_DEVICES'];state=RUN/f'results/worker_{role}_{gpu}.json'
    write_json(state,{'status':'running','gpu':gpu,'pid':os.getpid()})
    try:
        if role in ['speed','test_speed']:
            assert gpu=='2'
            name='candidate_pool' if role=='speed' else 'selected_policies'
            catalog=json.loads((RUN/f'configs/{name}.json').read_text())['policies']
            mode='calibration_speed' if role=='speed' else 'test_speed'
            for pid,job in catalog.items():
                marker=RUN/f'results/{mode}_complete/{pid}.json'
                if marker.exists():continue
                cleanup();model=checkpoint(pid)
                invoke([sys.executable,RUN/'code/evaluate.py','--checkpoint',model,'--policy',pid,'--mode',mode,'--scenarios',','.join(job['scenarios'])],pid+'_'+mode)
                write_json(marker,{'complete':True,'checkpoint':str(model),'scenarios':job['scenarios']});cleanup()
                print(mode,pid,'complete',flush=True)
            write_json(RUN/f'results/all_{mode}_complete.json',{'complete':True})
        else:
            mode=role;name='candidate_pool' if role=='validation' else 'selected_policies'
            catalog=json.loads((RUN/f'configs/{name}.json').read_text())['policies'];ready='calibration_speed' if role=='validation' else 'test_speed'
            while True:
                pending=0;claimed=False
                for pid,job in catalog.items():
                    marker=RUN/f'results/{mode}_complete/{pid}.json'
                    if marker.exists():continue
                    pending+=1;source=RUN/f'results/{ready}_complete/{pid}.json'
                    if not source.exists():continue
                    lock=RUN/f'results/locks/{mode}_{pid}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
                    with lock.open('a') as handle:
                        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:continue
                        if marker.exists():continue
                        model=json.loads(source.read_text())['checkpoint']
                        invoke([sys.executable,RUN/'code/evaluate.py','--checkpoint',model,'--policy',pid,'--mode',mode,'--scenarios',','.join(job['scenarios'])],pid+'_'+mode)
                        write_json(marker,{'complete':True,'gpu':gpu});claimed=True
                    cleanup()
                    print(mode,pid,'complete',flush=True);break
                if not pending:break
                if not claimed:time.sleep(5)
        write_json(state,{'status':'complete','gpu':gpu,'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','gpu':gpu,'error':repr(e)});raise


if __name__=='__main__':main()
