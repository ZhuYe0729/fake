"""Fixed-map Qwen transfers, sharing admission and timing locks."""
import argparse
import fcntl
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from launch import RUN,MODEL,SPEC,TEMP,checkpoint,invoke,read,write


def state(stage,**details):write(RUN/'results/worker.json',{'status':'running','stage':stage,**details})


def quality_worker():
    from resources import lease
    for pid in read(RUN/'configs/design.json')['policies']:
        marker=RUN/f'results/completed/quality/{pid}.json';lock=RUN/f'results/locks/{pid}.lock'
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a')as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            with lease()as gpu:
                model=checkpoint(pid,gpu)
                invoke([sys.executable,RUN/'code/quality_workloads.py','--checkpoint',model,'--policy',pid],'quality_'+pid,gpu)
                write(marker,{'complete':True,'gpu':gpu,'policy':pid})


def fingerprint():
    target=RUN/'reference/source_model_fingerprint.json'
    if target.exists():return
    files=[p for p in MODEL.iterdir()if p.is_file()and p.suffix in ['.safetensors','.json','.txt']]
    canonical=Path(SPEC['canonical_root'])
    files += [canonical/m/'model.pt'for m in ['sparse_bf16','sparse_nvfp4']]
    with(RUN.parents[1]/'night151/measurement.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_SH)
        digests={}
        for path in files:
            with path.open('rb')as source:digests[str(path)]=hashlib.file_digest(source,'sha256').hexdigest()
    write(target,{'files_sha256':digests,'scope':'Exact source and canonical bytes before own GPU outputs.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quality-worker',action='store_true');args=parser.parse_args()
    if args.quality_worker:return quality_worker()
    with(RUN/'results/driver.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not(RUN.parent/'037/results/final_audit.json').exists():state('wait037');time.sleep(10)
        replay=read(RUN.parent/'037/results/final_audit.json')
        assert replay['all_token_sequences_exact']and replay['all_nll_within_original_gate']
        fingerprint()
        design=read(RUN/'configs/design.json')
        if not(RUN/'results/quality_complete.json').exists():
            children=[]
            for i in range(3):
                log=(RUN/f'logs/quality_worker{i}_{time.time_ns()}.log').open('w')
                children.append((subprocess.Popen([sys.executable,__file__,'--quality-worker'],stdout=log,stderr=subprocess.STDOUT),log))
            while any(p.poll()is None for p,_ in children):
                state('quality',complete=len(list((RUN/'results/completed/quality').glob('*.json'))),total=len(design['policies']))
                time.sleep(10)
            assert all(p.returncode==0 for p,_ in children)
            for _,log in children:log.close()
            assert len(list((RUN/'results/completed/quality').glob('*.json')))==len(design['policies'])
            write(RUN/'results/quality_complete.json',{'complete':True})
        while not all((RUN.parent/r/'results/sweep_final_audit.json').exists()for r in ['038','039']):
            state('wait_phase_control_formal');time.sleep(10)
        from resources import lease
        for round_index,order in enumerate(design['rounds']):
            for index,label in enumerate(order):
                target=RUN/f'results/speed/round{round_index}/{label}.json'
                if target.exists():continue
                pid='uniform_dense_bf16'if label.endswith('_bracket_end')else label
                state('formal_speed',round=round_index,index=index,total=len(order),policy=pid)
                with lease(exclusive=True)as gpu:
                    model=checkpoint(pid,gpu);job=f'round{round_index}_{label}'
                    raw=RUN/f'results/measurements/{job}.json'
                    if not raw.exists():invoke([sys.executable,RUN/'code/benchmark_workloads.py','--checkpoint',model,'--policy',pid,'--label',job],job,gpu)
                    write(target,read(raw))
        write(RUN/'results/formal_complete.json',{'complete':True,'rounds':3})
        for path in(TEMP/'checkpoints').glob('*/model.safetensors'):
            write(RUN/f'results/cleanup/{path.parent.name}.json',{'path':str(path),'bytes':path.stat().st_size,'reason':'Own reproducible export; all measurements completed; metadata/logs/code/results retained.'});path.unlink()
        state('report')
        with(RUN.parents[1]/'night151/measurement.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            subprocess.run([sys.executable,RUN/'code/transfer_report.py'],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        write(RUN/'results/worker.json',{'status':'complete','stage':'audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)});raise
