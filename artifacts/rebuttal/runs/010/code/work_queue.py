"""Full downstream confirmation on otherwise idle cards, with policy locks."""
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

from common import MODEL,RUN,VLLM,WRAPPER,write_json

TEMP=Path('/dev/shm/daso_rebuttal_010_20261025')


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


def checkpoint(pid,origin):
    path=TEMP/'checkpoints'/pid
    if (path/'model.safetensors').exists() and (path/'phase_hetero_export_provenance.json').exists():return path
    while shutil.disk_usage('/dev/shm').free<24*1024**3:time.sleep(10)
    if path.exists():path.rename(path.with_name(pid+'_previous_'+str(time.time_ns())))
    policy=RUN/f'configs/policies/{pid}.json';content=policy.read_text()
    command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',policy,'--output-dir',path,'--cutlass-wrapper-path',WRAPPER]
    for method in ['sparse_bf16','sparse_nvfp4']:
        if method not in content:continue
        if pid.startswith('pruning_'):
            state=Path('/dev/shm/daso_rebuttal_007_20261025/canonical')/pid.removeprefix('pruning_')/'model.pt'
        else:state=Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt'
        command+=['--canonical-'+method.replace('_','-')+'-state',state]
    invoke(command,pid+'_export');directory=RUN/f'results/exports/{pid}';directory.mkdir(parents=True,exist_ok=True)
    for p in path.glob('*.json'):shutil.copy2(p,directory/p.name)
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['tasks','generation'],required=True);args=p.parse_args();kind=args.kind;gpu=os.environ['CUDA_VISIBLE_DEVICES']
    assert gpu in (['3','5'] if kind=='tasks' else ['0','6'])
    catalog=RUN/'configs/task_catalog.json'
    while not catalog.exists():time.sleep(10)
    if gpu=='5':
        marker=RUN.parent/'007/results/evaluation_worker.json'
        while not marker.exists() or json.loads(marker.read_text()).get('status')!='complete':time.sleep(10)
    if gpu in ['0','6']:
        marker=RUN.parent/('009/results/worker_speed.json' if gpu=='0' else '006/results/full_quality_worker.json')
        while not marker.exists() or json.loads(marker.read_text()).get('status')!='complete':time.sleep(10)
    jobs=[j for j in json.loads(catalog.read_text())['jobs'] if j['kind']==kind]
    worker=RUN/f'results/worker_{kind}_{gpu}.json';write_json(worker,{'status':'running','gpu':gpu,'pid':os.getpid()})
    try:
        while True:
            pending=0;worked=False
            for job in jobs:
                pid=job['policy_id'];done=RUN/f'results/complete/{job["id"]}.json'
                if done.exists():continue
                pending+=1;lock=RUN/f'results/claims/{pid}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
                with lock.open('a') as handle:
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if done.exists():continue
                    model=checkpoint(pid,job['origin']);print(kind,pid,'started',flush=True)
                    if kind=='tasks':
                        policy=RUN/f'configs/policies/{pid}.json';manifest=RUN/f'configs/manifest_{pid}.json'
                        write_json(manifest,{'model_path':str(MODEL),'policies':[{'label':pid,'kind':'ours','checkpoint':str(model),
                            'policy_json':str(policy),'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest()}]})
                        invoke([sys.executable,RUN/'code/run_workflow.py','tasks','--manifest',manifest,'--policy',pid,
                            '--output-root',RUN/'results/tasks','--vllm-root',VLLM,'--cutlass-wrapper-path',WRAPPER,'--audit'],pid+'_full_tasks')
                    else:
                        for dataset in ['cnn_dm_1000','dsum','IWSLT']:
                            target=RUN/f'results/generation/{pid}/{dataset}'
                            if (target/'metrics_0000_end.json').exists():continue
                            answers=target/'answers_0000_end.jsonl'
                            if answers.exists():answers.rename(answers.with_name('answers_failed_'+str(time.time_ns())+'.jsonl'))
                            invoke([sys.executable,RUN/'code/generate_tasks.py','--policy',pid,'--checkpoint',model,'--dataset',dataset],pid+'_full_'+dataset)
                    write_json(done,{'complete':True,'gpu':gpu});worked=True
                    alljobs=json.loads(catalog.read_text())['jobs']
                    if all((RUN/f'results/complete/{j["id"]}.json').exists() for j in alljobs if j['policy_id']==pid):
                        binary=model/'model.safetensors';size=binary.stat().st_size;binary.unlink()
                        write_json(RUN/f'results/exports/{pid}/binary_cleanup.json',{'bytes':size,'path':str(binary),
                            'reason':'all preregistered complete downstream jobs for this policy finished','preserved':'all code/log/CSV/result/document/metadata'})
                    print(kind,pid,'complete',flush=True)
                break
            if not pending:break
            if not worked:time.sleep(10)
        write_json(worker,{'status':'complete','gpu':gpu,'pid':os.getpid()})
    except Exception as e:write_json(worker,{'status':'failed','gpu':gpu,'error':repr(e)});raise


if __name__=='__main__':main()
