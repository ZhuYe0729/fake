"""Pruning controls follow the original-model core tables, using shared locks."""
import hashlib
import shutil
import sys
import time

from common import RUN, write_json
from launch import TEMP, checkpoint, cleanup, invoke, read
from resources import quality_gpus, speed_gpu


def state(stage,status='running',**details):write_json(RUN/'results/worker.json',{'status':status,'stage':stage,**details})


def main():
    assert TEMP.name=='daso_rebuttal_020_20261026'
    for rid,marker in [('017','final_audit'),('018','formal_complete')]:
        while not (RUN.parent/rid/f'results/{marker}.json').exists():
            state('source_core_table_and_other_speed','waiting',waiting_for=rid);time.sleep(10)
    assert read(RUN.parent/'017/results/final_audit.json')['verified']
    protection=RUN/'reference/protected_manifests.json'
    if not protection.exists():write_json(protection,{'017':hashlib.sha256((RUN.parent/'017/results/manifest.json').read_bytes()).hexdigest()})
    if not all((RUN/f'results/canonical_{method}.json').exists() for method in ['wanda','magnitude']):
        state('pruning','waiting')
        with quality_gpus(maximum=1,minimum=1) as cards:
            for method in ['wanda','magnitude']:
                marker=RUN/f'results/canonical_{method}.json'
                if marker.exists():continue
                state('pruning',method=method,gpu=cards[0])
                invoke([sys.executable,RUN/'code/prepare_pruning.py','--method',method],'prune_'+method,cards[0])
    for stage in ['confirmation','full_quality']:
        if not (RUN/f'results/{stage}_complete.json').exists():
            state(stage)
            invoke([sys.executable,RUN/'code/job_queue.py','--stage',stage],stage,6)
    state('formal_speed','waiting')
    with speed_gpu():
        if not (RUN/'results/formal_complete.json').exists():
            for campaign,order in enumerate(read(RUN/'configs/design.json')['rounds']):
                for index,label in enumerate(order):
                    target=RUN/f'results/speed/round{campaign}/{label}.json'
                    if target.exists():continue
                    pid='uniform_dense_bf16' if label.endswith('_bracket_end') else label
                    state('formal_speed',campaign=campaign,index=index,policy=pid)
                    model=checkpoint(pid,0)
                    job=f'round{campaign}_{label}'
                    raw=RUN/f'results/measurements/{job}.json'
                    if not raw.exists():invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',model,
                        '--policy',pid,'--mode','arena_graph','--label',job,'--outputs','1','64','--iterations','8'],job,0)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(raw,target)
                    cleanup(model,job)
            write_json(RUN/'results/formal_complete.json',{'complete':True,'campaigns':3,'processes':12})
        # Hashing the full pruning states also stays outside other speed runs.
        state('report')
        invoke([sys.executable,RUN/'code/report.py'],'complete_report',6)
        state('audit')
        invoke([sys.executable,RUN/'code/audit.py'],'audit',6)
    state('audited','complete')


if __name__=='__main__':
    try:main()
    except Exception as error:
        old=read(RUN/'results/worker.json') if (RUN/'results/worker.json').exists() else {}
        write_json(RUN/'results/worker.json',{**old,'status':'failed','error':repr(error)})
        raise
