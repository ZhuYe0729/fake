"""Run the frozen small generation comparison on idle, claimed GPUs."""
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

RUN=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def state(stage,status='running',**details):
    write(RUN/'results/worker.json',{'stage':stage,'status':status,**details})


def invoke(argv,label,gpu):
    first=gpu*8
    command=['numactl','--all',f'--physcpubind={first}-{first+7}',f'--membind={0 if gpu<4 else 1}',*map(str,argv)]
    log=RUN/f'logs/{label}_{time.time_ns()}.log'
    write(log.with_suffix('.command.json'),{'command':command,'gpu':gpu,
        'design_sha256':sha(RUN/'configs/design.json'),
        'code_sha256':{p.name:sha(p) for p in (RUN/'code').glob('*.py')}})
    start=time.perf_counter()
    with log.open('w') as handle:
        result=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=handle,stderr=subprocess.STDOUT)
    write(log.with_suffix('.completion.json'),{'seconds':time.perf_counter()-start,'returncode':result.returncode})
    assert result.returncode==0,str(log)


def worker(gpu):
    from common import MODEL,WRAPPER
    design=read(RUN/'configs/design.json')
    for pid in design['policies']:
        marker=RUN/f'results/completed/{pid}.json'
        if marker.exists():continue
        lock=RUN/f'results/locks/{pid}.lock';lock.parent.mkdir(exist_ok=True)
        with lock.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            assert sha(RUN/f'configs/policies/{pid}.json')==design['policies'][pid]['policy_sha256']
            model=RUN/f'data/checkpoints/{pid}'
            if not (model/'model.safetensors').exists():
                assert shutil.disk_usage(RUN).free>90*1024**3
                if model.exists():model.rename(model.with_name(pid+'_incomplete_'+str(time.time_ns())))
                with (RUN.parents[1]/'night151/export.lock').open('a') as export:
                    fcntl.flock(export,fcntl.LOCK_EX)
                    invoke([sys.executable,RUN/'code/export_model.py','--model-path',MODEL,
                        '--policy-json',RUN/f'configs/policies/{pid}.json','--output-dir',model,
                        '--cutlass-wrapper-path',WRAPPER],'export_'+pid,gpu)
            for dataset in design['datasets']:
                folder=RUN/f'results/generation/{pid}/{dataset}'
                if (folder/'metrics_0000_end.json').exists():continue
                partial=folder/'answers_0000_end.jsonl'
                if partial.exists():partial.rename(partial.with_name('answers_failed_'+str(time.time_ns())+'.jsonl'))
                invoke([sys.executable,RUN/'code/generate_tasks.py','--policy',pid,
                    '--checkpoint',model,'--dataset',dataset],pid+'_'+dataset,gpu)
            write(marker,{'complete':True,'policy':pid,'gpu':gpu})
    full=design.get('full_candidate_generation')
    if full:
        pid=full['policy'];model=RUN/f'data/checkpoints/{pid}'
        assert (model/'model.safetensors').exists()
        for dataset in design['datasets']:
            lock=RUN/f'results/full_locks/{dataset}.lock';lock.parent.mkdir(exist_ok=True)
            with lock.open('a') as handle:
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                target=RUN/f'full/results/generation/{pid}/{dataset}'
                if (target/'metrics_0000_end.json').exists():continue
                partial=target/'answers_0000_end.jsonl'
                if partial.exists():partial.rename(partial.with_name('answers_failed_'+str(time.time_ns())+'.jsonl'))
                invoke([sys.executable,RUN/'code/generate_full.py','--policy',pid,
                    '--checkpoint',model,'--dataset',dataset],'full_'+pid+'_'+dataset,gpu)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int);args=parser.parse_args()
    if args.gpu is not None:worker(args.gpu);return
    state('measurement_lock','waiting',reason='Existing quiet speed campaigns retain priority; no GPU work before shared lock.')
    from resources import quality_gpus
    with quality_gpus(maximum=2,minimum=1) as cards:
        children=[]
        for gpu in cards:
            log=(RUN/f'logs/quality_worker_gpu{gpu}_{time.time_ns()}.log').open('w')
            child=subprocess.Popen([sys.executable,__file__,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT)
            children.append((child,log))
        while any(child.poll() is None for child,_ in children):
            state('pilot_generation',gpus=cards,policies_complete=len(list((RUN/'results/completed').glob('*.json'))),
                generation_tasks_complete=len(list((RUN/'results/generation').glob('*/*/metrics_0000_end.json'))),total_generation_tasks=12,
                workers=[{'pid':child.pid,'returncode':child.poll()} for child,_ in children])
            time.sleep(10)
        codes=[child.returncode for child,_ in children]
        for _,log in children:log.close()
        assert len(list((RUN/'results/completed').glob('*.json')))==4,codes
    state('pilot_report')
    subprocess.run([sys.executable,RUN/'code/report.py'],check=True)
    state('pilot_complete','complete',generation_tasks=12)


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('failed','failed',error=repr(error));raise
