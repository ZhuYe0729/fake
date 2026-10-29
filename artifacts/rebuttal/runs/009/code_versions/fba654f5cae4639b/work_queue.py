"""Same-card speed and independent quality for all fixed HAWQ controls."""
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

TEMP=Path('/dev/shm/daso_rebuttal_009_20261025')


def invoke(command,label):
    log=RUN/f'logs/{label}_{time.time_ns()}.log';files=sorted((RUN/'code').glob('*.py'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot=RUN/'code_versions'/version;snapshot.mkdir(parents=True,exist_ok=True)
    for p in files:
        if not (snapshot/p.name).exists():shutil.copy2(p,snapshot/p.name)
    write_json(log.with_suffix('.command.json'),{'command':list(map(str,command)),'code_version':version,'code_hashes':hashes,'gpu':os.environ['CUDA_VISIBLE_DEVICES']})
    start=time.perf_counter()
    with log.open('w') as f:result=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
    if result.returncode:raise RuntimeError(f'exit {result.returncode}: {log}')


def cleanup():
    for path in (TEMP/'checkpoints').glob('*/model.safetensors'):
        pid=path.parent.name
        if not (RUN/f'results/speed_complete/{pid}.json').exists() or not (RUN/f'results/quality_complete/{pid}.json').exists():continue
        try:size=path.stat().st_size;path.unlink()
        except FileNotFoundError:continue
        write_json(RUN/f'results/exports/{pid}/binary_cleanup.json',{'bytes':size,'path':str(path),
            'reason':'all fixed speed and test-quality evaluation complete','preserved':'all source/log/metadata/CSV/results/documents'})


def checkpoint(pid):
    if pid=='uniform_dense_bf16':return Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/pid
    path=TEMP/'checkpoints'/pid
    if (path/'model.safetensors').exists():return path
    while shutil.disk_usage('/dev/shm').free<32*1024**3:cleanup();time.sleep(5)
    command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',RUN/f'configs/policies/{pid}.json',
        '--output-dir',path,'--cutlass-wrapper-path',WRAPPER]
    content=(RUN/f'configs/policies/{pid}.json').read_text()
    for method in ['sparse_bf16','sparse_nvfp4']:
        if method in content:command+=['--canonical-'+method.replace('_','-')+'-state',Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt']
    invoke(command,pid+'_export');directory=RUN/f'results/exports/{pid}';directory.mkdir(parents=True,exist_ok=True)
    for p in path.glob('*.json'):shutil.copy2(p,directory/p.name)
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['speed','quality'],required=True);args=p.parse_args()
    role=args.role;assert os.environ['CUDA_VISIBLE_DEVICES']==('0' if role=='speed' else '1')
    assert (RUN.parent/'005/configs/selected_policies.json').exists()
    catalog=json.loads((RUN/'configs/candidate_pool.json').read_text())['policies'];state=RUN/f'results/worker_{role}.json'
    write_json(state,{'status':'running','gpu':os.environ['CUDA_VISIBLE_DEVICES'],'pid':os.getpid()})
    try:
        while True:
            pending=0;progress=False
            for pid,job in catalog.items():
                done=RUN/f'results/{role}_complete/{pid}.json'
                if done.exists():continue
                pending+=1
                if role=='quality':
                    ready=RUN/f'results/speed_complete/{pid}.json'
                    if not ready.exists():continue
                    model=Path(json.loads(ready.read_text())['checkpoint'])
                else:cleanup();model=checkpoint(pid)
                invoke([sys.executable,RUN/'code/evaluate.py','--checkpoint',model,'--policy',pid,
                    '--mode','calibration_speed' if role=='speed' else 'test','--scenarios',','.join(job['scenarios'])],pid+'_'+role)
                write_json(done,{'complete':True,'checkpoint':str(model)});cleanup();progress=True
                print(role,pid,'complete',flush=True)
                if role=='quality':break
            if not pending:break
            if not progress:time.sleep(5)
        write_json(state,{'status':'complete','gpu':os.environ['CUDA_VISIBLE_DEVICES'],'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
