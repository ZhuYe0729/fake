"""Replay audited034/035 baseline weights independently of optional036 timing."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]/'night161';LOCKS=RUN.parents[1]/'night151'
read=lambda p:json.loads(p.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


@contextlib.contextmanager
def lease():
    sys.path.insert(0,str(RUN.parent/'034/code'))
    from telemetry import nvml
    while True:
        phase=(LOCKS/'measurement.lock').open('a')
        try:fcntl.flock(phase,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:
            phase.close();time.sleep(5);continue
        claim=None
        admission=read(ROOT/'gpu_admission.json')['cards']
        for gpu in admission:
            device=nvml.nvmlDeviceGetHandleByIndex(gpu)
            if nvml.nvmlDeviceGetMemoryInfo(device).used>=512*1024**2:continue
            handle=(LOCKS/f'gpu_locks/gpu{gpu}.lock').open('a')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close();continue
            if nvml.nvmlDeviceGetMemoryInfo(device).used>=512*1024**2:
                handle.close();continue
            claim=handle;break
        if claim:
            try:yield gpu
            finally:claim.close();phase.close()
            return
        phase.close();time.sleep(5)


def invoke(command,label,gpu):
    argv=['numactl','--all',f'--physcpubind={gpu*8}-{gpu*8+7}',f'--membind={0 if gpu<4 else 1}',*map(str,command)]
    log=RUN/f'logs/{label}_{time.time_ns()}.log';started=time.time()
    write(log.with_suffix('.command.json'),{'command':argv,'gpu':gpu,'started_at_unix':started})
    with log.open('w')as handle:
        code=subprocess.run(argv,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=handle,stderr=subprocess.STDOUT).returncode
    write(log.with_suffix('.completion.json'),{'seconds':time.time()-started,'returncode':code})
    assert code==0,str(log)


def worker():
    for job in read(RUN/'configs/jobs.json'):
        label=job['source_run']+'_'+job['policy'];marker=RUN/f'results/completed/{label}.json'
        if marker.exists():continue
        lock=RUN/f'results/locks/{label}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a')as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            checkpoint=ROOT/'checkpoints'/job['source_run']/job['policy']
            assert(checkpoint/'model.safetensors').exists(),str(checkpoint)
            target=RUN/'models'/job['model_id']
            source=RUN.parent/job['source_run']
            files=[source/'code'/n for n in ['common.py','eightbit.py','export_model.py','runtime_extensions.py',
                'arena_runtime.py','graph_bindings.py','admission_check.py','generate_tasks.py','score_nll.py']]
            files += [source/'configs/model.json',source/'configs/generation_protocol.json',
                source/'data/activation_scales.pt',checkpoint/'config.json',source/'reference/source_model_fingerprint.json']
            identity={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
            before=read(checkpoint/'config.json')['quantization_config']['method_map']
            assert before==read(source/f'configs/policies/{job["policy"]}.json')['method_map']
            write(RUN/f'results/source_identity/{label}.json',{'files_sha256':identity,
                'checkpoint_inode':(checkpoint/'model.safetensors').stat().st_ino,
                'checkpoint_bytes':(checkpoint/'model.safetensors').stat().st_size,'effective_map_equal':True})
            with lease()as gpu:
                for kind,task in [('nll',None),*[("generation",d)for d in ['cnn_dm_1000','dsum','IWSLT']]]:
                    finished=target/f'results/confirmation/{job["policy"]}/prefill_decode.json'if kind=='nll'else target/f'results/generation/{job["policy"]}/{task}/metrics_0000_128.json'
                    if finished.exists():continue
                    command=[sys.executable,RUN/'code/replay.py','--source-run',job['source_run'],
                        '--root',target,'--checkpoint',checkpoint,'--policy',job['policy'],'--kind',kind]
                    if task:command+=['--dataset',task]
                    invoke(command,label+'_'+kind+('_'+task if task else''),gpu)
                for path,digest in identity.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
                write(marker,{'complete':True,'gpu':gpu,**job})


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:return worker()
    with(RUN/'results/driver.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not all((RUN.parent/r/'results/sweep_final_audit.json').exists()for r in ['034','035']):
            write(RUN/'results/worker.json',{'status':'waiting','stage':'original_sweep_audits'});time.sleep(10)
        children=[]
        for i in range(6):
            log=(RUN/f'logs/worker{i}_{time.time_ns()}.log').open('w')
            child=subprocess.Popen([sys.executable,__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT);children.append((child,log))
        while any(p.poll()is None for p,_ in children):
            write(RUN/'results/worker.json',{'status':'running','stage':'baseline_replay','complete':len(list((RUN/'results/completed').glob('*.json'))),'total':6});time.sleep(10)
        assert all(p.returncode==0 for p,_ in children)
        for _,log in children:log.close()
        subprocess.run([sys.executable,RUN/'code/report.py'],check=True)
        write(RUN/'results/worker.json',{'status':'complete','stage':'audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)});raise
