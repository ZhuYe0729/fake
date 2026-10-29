"""Run frozen mixed cases after all higher-priority campaigns finish."""
import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import time

from launch import RUN,TEMP,checkpoint,invoke,read,write


def sha(path):
    with path.open('rb')as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def state(stage,**details):write(RUN/'results/worker.json',{'status':'running','stage':stage,**details})


def worker():
    from baseline_resources import lease
    for pid in read(RUN/'configs/design.json')['policies']:
        marker=RUN/f'results/completed/{pid}.json';lock=RUN/f'results/locks/{pid}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a')as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            if not(RUN/f'results/quality/quality_{pid}.json').exists():
                with lease()as gpu:
                    model=checkpoint(pid,gpu)
                    invoke([sys.executable,RUN/'code/benchmark_mixed.py','--checkpoint',model,'--policy',pid,
                        '--quality','--label','quality_'+pid],'quality_'+pid,gpu)
            write(marker,{'complete':True,'policy':pid})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:return worker()
    with(RUN/'results/driver.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not all((RUN.parent/r/'results/final_audit.json').exists()for r in ['044','045','046']):
            state('wait_fresh_quality');time.sleep(10)
        while not(RUN.parent/'048/results/final_audit.json').exists():state('wait_repeat048');time.sleep(10)
        assert read(RUN.parent/'048/results/final_audit.json')['verified']
        for r in ['044','045','046']:assert read(RUN.parent/r/'results/final_audit.json')['verified']
        from baseline_resources import lease
        source=RUN.parent/'040/reference/source_model_fingerprint.json';target=RUN/'reference/source_model_fingerprint.json'
        if not target.exists():
            with(RUN.parents[1]/'night151/measurement.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_SH)
                for path,digest in read(source)['files_sha256'].items():
                    from pathlib import Path
                    assert sha(Path(path))==digest,path
            shutil.copy2(source,target)
        gate=RUN/'results/bf16_noop_gate.json'
        if not gate.exists():
            pid='uniform_dense_bf16'
            for noop in [False,True]:
                label='quality_bf16_noop'if noop else'quality_'+pid
                if(RUN/f'results/quality/{label}.json').exists():continue
                state('bf16_noop_gate',noop=noop)
                with lease()as gpu:
                    model=checkpoint(pid,gpu)
                    argv=[sys.executable,RUN/'code/benchmark_mixed.py','--checkpoint',model,'--policy',pid,'--quality','--label',label]
                    invoke(argv+(['--noop']if noop else[]),label,gpu)
            first=read(RUN/f'results/quality/quality_{pid}.json');second=read(RUN/'results/quality/quality_bf16_noop.json')
            difference=max(abs(a['avg_nll']-b['avg_nll'])for x,y in zip(first['cases'],second['cases'],strict=True)
                for a,b in zip(x['blocks'],y['blocks'],strict=True))
            assert difference<=5e-6,difference
            write(gate,{'verified':True,'max_window_nll_difference':difference,'threshold':5e-6,
                'routing_sha256':sha(RUN/f'results/quality/quality_{pid}.json'),'noop_sha256':sha(RUN/'results/quality/quality_bf16_noop.json')})
        if not(RUN/'results/quality_complete.json').exists():
            children=[]
            for i in range(3):
                log=(RUN/f'logs/worker{i}_{time.time_ns()}.log').open('w')
                children.append((subprocess.Popen([sys.executable,__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT),log))
            while any(p.poll()is None for p,_ in children):
                state('quality',complete=len(list((RUN/'results/completed').glob('*.json'))),total=9);time.sleep(10)
            assert all(p.returncode==0 for p,_ in children)
            for _,log in children:log.close()
            assert len(list((RUN/'results/completed').glob('*.json')))==9
            write(RUN/'results/quality_complete.json',{'complete':True})
        for ri,order in enumerate(read(RUN/'configs/design.json')['rounds']):
            for index,label in enumerate(order):
                tag=f'round{ri}_{label}'
                if(RUN/f'results/measurements/{tag}.json').exists():continue
                pid='uniform_dense_bf16'if label.endswith('_bracket_end')else label
                state('formal_speed',round=ri,index=index,total=len(order),policy=pid)
                with lease(exclusive=True)as gpu:
                    model=checkpoint(pid,gpu)
                    invoke([sys.executable,RUN/'code/benchmark_mixed.py','--checkpoint',model,'--policy',pid,'--label',tag],tag,gpu)
        state('report')
        subprocess.run([sys.executable,RUN/'code/mixed_report.py'],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        for path in(TEMP/'checkpoints').glob('*/model.safetensors'):
            write(RUN/f'results/cleanup/{path.parent.name}.json',{'path':str(path),'bytes':path.stat().st_size,
                'reason':'Own reproducible weights, all jobs and audit complete; code/logs/CSV/docs/captures kept.'});path.unlink()
        write(RUN/'results/worker.json',{'status':'complete','stage':'audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)});raise
