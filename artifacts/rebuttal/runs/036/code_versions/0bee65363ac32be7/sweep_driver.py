"""Run every frozen Llama point; existing MC/Prefill-only results stay unchanged."""
import argparse
import fcntl
import os
import random
import subprocess
import sys
import time

from launch import RUN,SPEC,TEMP,checkpoint,invoke,read,write


def state(stage,**details):
    write(RUN/'results/sweep_worker.json',{'status':'running','stage':stage,**details})


def worker(stage):
    from sweep_resources import lease
    for item in read(RUN/f'configs/sweep_{stage}_jobs.json')['jobs']:
        pid=item['policy'];label=pid+('_'+item['dataset']if stage=='generation'else'')
        marker=RUN/f'results/sweep_completed/{stage}/{label}.json'
        if marker.exists():continue
        path=RUN/f'results/sweep_locks/{stage}_{label}.lock';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a')as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            with lease()as gpu:
                model=checkpoint(pid,gpu)
                if stage=='evaluation':
                    command=[sys.executable,RUN/'code/score_nll.py','--checkpoint',model,'--policy',pid,
                             '--split','confirmation','--scenarios','prefill_decode']
                else:
                    command=[sys.executable,RUN/'code/generate_tasks.py','--checkpoint',model,
                             '--policy',pid,'--dataset',item['dataset']]
                invoke(command,'sweep_'+stage+'_'+label,gpu)
                write(marker,{'complete':True,'gpu':gpu,**item})


def quality(stage):
    jobs=read(RUN/f'configs/sweep_{stage}_jobs.json')['jobs'];children=[]
    for i in range(3):
        log=(RUN/f'logs/sweep_{stage}_worker{i}_{time.time_ns()}.log').open('w')
        child=subprocess.Popen([sys.executable,__file__,'--worker',stage],stdout=log,stderr=subprocess.STDOUT)
        children.append((child,log))
    while any(p.poll()is None for p,_ in children):
        state(stage,complete=len(list((RUN/f'results/sweep_completed/{stage}').glob('*.json'))),total=len(jobs),
              workers=[{'pid':p.pid,'returncode':p.poll()}for p,_ in children])
        time.sleep(10)
    assert all(p.returncode==0 for p,_ in children),[(p.pid,p.returncode)for p,_ in children]
    for _,log in children:log.close()
    assert len(list((RUN/f'results/sweep_completed/{stage}').glob('*.json')))==len(jobs)
    write(RUN/f'results/sweep_{stage}_complete.json',{'complete':True,'jobs':len(jobs)})


def formal():
    from sweep_resources import lease
    path=RUN/'configs/sweep_formal_rounds.json'
    if not path.exists():
        pids=list(read(RUN/'configs/policy_registry.json'));orders=[]
        for i in range(3):
            middle=[p for p in pids if p!='uniform_dense_bf16']
            random.Random(160034+i).shuffle(middle)
            orders.append(['uniform_dense_bf16',*middle,'uniform_dense_bf16_bracket_end'])
        write(path,{'rounds':orders})
    for i,order in enumerate(read(path)['rounds']):
        for j,label in enumerate(order):
            target=f'sweep_formal_round{i}_{label}'
            if(RUN/f'results/measurements/{target}.json').exists():continue
            state('formal_speed',round=i,complete=j,total=len(order),policy=label)
            pid='uniform_dense_bf16'if label.endswith('_bracket_end')else label
            with lease(exclusive=True)as gpu:
                model=checkpoint(pid,gpu)
                invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',pid,
                    '--mode','arena_graph','--label',target,'--outputs','64','--iterations','5',
                    '--audit-weights','--input-file',RUN/'configs/speed_inputs.json'],target,gpu)
    write(RUN/'results/sweep_formal_complete.json',{'complete':True,'rounds':3})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['evaluation','generation']);args=parser.parse_args()
    if args.worker:return worker(args.worker)
    with(RUN/'results/sweep_driver.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(RUN/'results/sweep_design_checks.json')['verified']
        while SPEC['source_run']=='018'and not(RUN/'results/source_model_complete.json').exists():
            state('source_model_restore');time.sleep(10)
        for stage in ['evaluation','generation']:
            if not(RUN/f'results/sweep_{stage}_complete.json').exists():quality(stage)
        while not all((RUN.parent/r/'results/sweep_final_audit.json').exists() for r in ['034','035']):
            state('waiting_original_campaigns');time.sleep(10)
        if not(RUN/'results/sweep_formal_complete.json').exists():formal()
        for path in (TEMP/'checkpoints').glob('*/model.safetensors'):
            write(RUN/f'results/sweep_cleanup/{path.parent.name}.json',{'path':str(path),'bytes':path.stat().st_size,
                'reason':'Own reproducible export, all jobs finished; code/logs/metadata/results retained.'})
            path.unlink()
        state('report')
        # Shared audit lock prevents full source hashing from perturbing another formal campaign.
        lock=RUN.parents[1]/'night151/measurement.lock'
        with lock.open('r')as phase:
            fcntl.flock(phase,fcntl.LOCK_SH)
            subprocess.run([sys.executable,RUN/'code/sweep_report.py'],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        write(RUN/'results/sweep_worker.json',{'status':'complete','stage':'audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/f'results/sweep_failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)})
        raise
