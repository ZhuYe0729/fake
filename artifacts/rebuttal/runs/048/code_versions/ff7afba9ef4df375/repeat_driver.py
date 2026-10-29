"""Current-protocol complete generation and matched timing for all ten maps."""
import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from launch import RUN,TEMP,checkpoint,invoke,read,write


def state(stage,**details):write(RUN/'results/worker.json',{'status':'running','stage':stage,**details})


def worker():
    from resources import lease
    for job in read(RUN/'configs/design.json')['jobs']:
        pid,task=job['policy'],job['dataset'];label=pid+'_'+task
        marker=RUN/f'results/completed/{label}.json';lock=RUN/f'results/job_locks/{label}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a')as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            folder=RUN/f'results/generation/{pid}/{task}';metric=folder/'metrics_0000_end.json'
            if not metric.exists():
                if folder.exists():
                    target=RUN/f'results/interrupted_generation/{pid}/{task}_{time.time_ns()}';target.parent.mkdir(parents=True,exist_ok=True)
                    folder.rename(target);write(target/'RETRY.json',{'reason':'Incomplete own attempt; all answers/metadata kept before fresh full job.'})
                with lease()as gpu:
                    model=checkpoint(pid,gpu)
                    invoke([sys.executable,RUN/'code/generate_tasks.py','--checkpoint',model,'--policy',pid,'--dataset',task],label,gpu)
            write(marker,{'complete':True,**job})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:return worker()
    with(RUN/'results/driver.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not all((RUN.parent/r/'results/final_audit.json').exists()for r in ['044','045','046']):
            state('wait_fresh_quality');time.sleep(10)
        for r in ['044','045','046']:assert read(RUN.parent/r/'results/final_audit.json')['verified']
        design=read(RUN/'configs/design.json')
        for name,digest in design['numeric_sha256'].items():assert hashlib.sha256((RUN/'code'/name).read_bytes()).hexdigest()==digest
        if not(RUN/'results/source_identity_check.json').exists():
            with(RUN.parents[1]/'night151/measurement.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_SH)
                for path,digest in read(RUN/'reference/source_model_fingerprint.json')['files_sha256'].items():
                    with Path(path).open('rb')as source:assert hashlib.file_digest(source,'sha256').hexdigest()==digest,path
            write(RUN/'results/source_identity_check.json',{'verified':True,'time_unix':time.time()})
        if not(RUN/'results/quality_complete.json').exists():
            children=[]
            for i in range(3):
                log=(RUN/f'logs/worker{i}_{time.time_ns()}.log').open('w')
                children.append((subprocess.Popen([sys.executable,__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT),log))
            while any(p.poll()is None for p,_ in children):
                state('generation',complete=len(list((RUN/'results/completed').glob('*.json'))),total=30);time.sleep(10)
            assert all(p.returncode==0 for p,_ in children)
            for _,log in children:log.close()
            assert len(list((RUN/'results/completed').glob('*.json')))==30
            write(RUN/'results/quality_complete.json',{'complete':True})
        from resources import lease
        for ri,order in enumerate(design['rounds']):
            for index,label in enumerate(order):
                tag=f'repeat_round{ri}_{label}'
                if(RUN/f'results/measurements/{tag}.json').exists():continue
                pid='uniform_dense_bf16'if label.endswith('_bracket_end')else label
                state('formal_speed',round=ri,index=index,total=len(order),policy=pid)
                with lease(exclusive=True)as gpu:
                    model=checkpoint(pid,gpu)
                    invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',pid,'--mode','arena_graph',
                        '--label',tag,'--outputs','1','64','--iterations','5','--audit-weights','--input-file',RUN/'configs/speed_inputs.json'],tag,gpu)
        #047 uses exactly the same representations; hand off hardlinks before
        #publishing048's audit, so it cannot start reading a partial handoff.
        future=RUN.parent/'047';future_temp=Path(read(future/'configs/model.json')['temp'])
        assert read(future/'configs/model.json')['model_path']==read(RUN/'configs/model.json')['model_path']
        assert hashlib.sha256((future/'data/activation_scales.pt').read_bytes()).digest()==hashlib.sha256((RUN/'data/activation_scales.pt').read_bytes()).digest()
        for name in ['common.py','eightbit.py','export_model.py','runtime_extensions.py']:
            assert(future/'code'/name).read_bytes()==(RUN/'code'/name).read_bytes()
        with(RUN.parents[1]/'night151/export.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for pid in read(future/'configs/design.json')['policies']:
                src=TEMP/'checkpoints'/pid;dst=future_temp/'checkpoints'/pid
                assert read(src/'config.json')['quantization_config']['method_map']==read(future/f'configs/policies/{pid}.json')['method_map']
                dst.mkdir(parents=True,exist_ok=True)
                for path in src.iterdir():
                    if path.is_file()and path.name!='model.safetensors':shutil.copy2(path,dst/path.name)
                target=dst/'model.safetensors'
                if not target.exists():os.link(src/'model.safetensors',target)
                stat=target.stat();assert stat.st_ino==(src/'model.safetensors').stat().st_ino
                write(future/f'results/checkpoint_reuse/{pid}.json',{'source':str(src),'target':str(dst),'exact_hardlink':True,
                    'inode':stat.st_ino,'bytes':stat.st_size,'physical_copy_bytes':0,'representation_source_run':'048'})
        state('report')
        with(RUN.parents[1]/'night151/measurement.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            subprocess.run([sys.executable,RUN/'code/repeat_report.py'],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        for path in(TEMP/'checkpoints').glob('*/model.safetensors'):
            write(RUN/f'results/cleanup/{path.parent.name}.json',{'path':str(path),'bytes':path.stat().st_size,
                'remaining_hardlinks_after_unlink':path.stat().st_nlink-1,'physical_bytes_released_now':path.stat().st_size if path.stat().st_nlink==1 else 0,
                'reason':'Own reproducible export; complete generation, formal timing and report audited; other files kept.'});path.unlink()
        write(RUN/'results/worker.json',{'status':'complete','stage':'audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)});raise
