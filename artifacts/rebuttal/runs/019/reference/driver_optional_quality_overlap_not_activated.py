"""Run supplemental fixed-map loads after both original-model core tables."""
import argparse
import fcntl
import hashlib
import shutil
import subprocess
import sys
import time

from common import RUN, write_json
from launch import checkpoint, cleanup, invoke, read
from resources import quality_gpus, speed_gpu, ROOT


def state(stage, status='running', **details):
    write_json(RUN/'results/worker.json',{'status':status,'stage':stage,**details})


def export(pid,gpu):
    with (ROOT/'export.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        while shutil.disk_usage('/dev/shm').free<26*1024**3:time.sleep(10)
        return checkpoint(pid,gpu)


def quality_worker(gpu):
    design = read(RUN/'configs/design.json')
    for pid in sorted(design['policies']):
        marker = RUN/f'results/completed/quality/{pid}.json'
        if marker.exists():continue
        lock = RUN/f'results/locks/{pid}.lock'
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a') as claim:
            try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            for case in design['workloads']:
                partial = RUN/f'results/captures/{pid}/{case["label"]}'
                if partial.exists() and not (RUN/f'results/quality/{pid}/{case["label"]}.json').exists():
                    partial.rename(partial.with_name(case['label']+'_failed_'+str(time.time_ns())))
            model = export(pid,gpu)
            invoke([sys.executable,RUN/'code/quality_workloads.py','--checkpoint',model,'--policy',pid], 'quality_'+pid,gpu)
            write_json(marker,{'complete':True,'policy':pid,'gpu':gpu})
            cleanup(model,'quality_'+pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality-worker',type=int)
    args = parser.parse_args()
    if args.quality_worker is not None:
        quality_worker(args.quality_worker)
        return
    design = read(RUN/'configs/design.json')
    # Core tables retain priority; no background supplemental work blocks them.
    for rid in ['017','018']:
        while not (RUN.parent/rid/'results/formal_complete.json').exists():
            other = read(RUN.parent/rid/'results/worker.json')
            state('original_models_core_speed','waiting',waiting_for=rid,other_status=other)
            time.sleep(10)
        assert read(RUN.parent/rid/'results/formal_complete.json')['complete']
    while not any((RUN.parent/r/'results/final_audit.json').exists() for r in ['017','018']):
        state('first_original_model_core_table','waiting')
        time.sleep(10)
    if not (RUN/'results/quality_complete.json').exists():
        state('workload_quality','waiting')
        with quality_gpus() as cards:
            children=[]
            for gpu in cards:
                log=(RUN/f'logs/quality_worker_gpu{gpu}_{time.time_ns()}.log').open('w')
                child=subprocess.Popen([sys.executable,__file__,'--quality-worker',str(gpu)],stdout=log,stderr=subprocess.STDOUT)
                children.append((child,log))
            while any(child.poll() is None for child,_ in children):
                count=len(list((RUN/'results/completed/quality').glob('*.json')))
                state('workload_quality',complete=count,total=len(design['policies']),gpus=cards)
                time.sleep(10)
            codes=[child.returncode for child,_ in children]
            for _,log in children:log.close()
            pending=[pid for pid in design['policies'] if not (RUN/f'results/completed/quality/{pid}.json').exists()]
            assert not pending,{'pending':pending,'worker_exit_codes':codes}
            write_json(RUN/'results/quality_complete.json',{'complete':True,'policies':len(design['policies']),'worker_exit_codes':codes})
    if not (RUN/'results/formal_complete.json').exists():
        state('workload_speed','waiting')
        with speed_gpu():
            for campaign,order in enumerate(design['rounds']):
                for index,label in enumerate(order):
                    target=RUN/f'results/speed/round{campaign}/{label}.json'
                    if target.exists():continue
                    pid='uniform_dense_bf16' if label.endswith('_bracket_end') else label
                    state('workload_speed',campaign=campaign,index=index,total=len(order),policy=pid)
                    model=export(pid,0)
                    job=f'round{campaign}_{label}'
                    raw=RUN/f'results/measurements/{job}.json'
                    if not raw.exists():
                        invoke([sys.executable,RUN/'code/benchmark_workloads.py','--checkpoint',model,'--policy',pid,'--label',job],job,0)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(raw,target)
                    cleanup(model,job)
            write_json(RUN/'results/formal_complete.json',{'complete':True,'campaigns':3,'processes':sum(map(len,design['rounds']))})
    state('report')
    invoke([sys.executable,RUN/'code/report.py'],'complete_report',6)
    state('audit')
    invoke([sys.executable,RUN/'code/audit.py'],'audit',6)
    state('audited','complete')


if __name__ == '__main__':
    try:main()
    except Exception as error:
        old=read(RUN/'results/worker.json') if (RUN/'results/worker.json').exists() else {}
        write_json(RUN/'results/worker.json',{**old,'status':'failed','error':repr(error)})
        raise
