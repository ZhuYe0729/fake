"""Use idle cards for the six independent generation tasks while MC continues."""
import argparse
import fcntl
import subprocess
import sys
import time

from common import RUN,write_json
from launch import TEMP,invoke

JOBS=[(pid,dataset) for dataset in ['cnn_dm_1000','dsum','IWSLT']
      for pid in ['uniform_wanda','uniform_magnitude']]


def worker(gpu):
    for pid,dataset in JOBS:
        target=RUN/f'results/generation/{pid}/{dataset}/metrics_0000_end.json'
        if target.exists():continue
        claim=RUN/f'results/parallel_generation_claims/{pid}_{dataset}.lock'
        claim.parent.mkdir(parents=True,exist_ok=True)
        with claim.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if target.exists():continue
            model=TEMP/'checkpoints'/pid
            while not (model/'model.safetensors').exists():
                if target.exists():break
                time.sleep(5)
            if target.exists():continue
            invoke([sys.executable,RUN/'code/generate_tasks.py','--checkpoint',model,
                '--policy',pid,'--dataset',dataset],f'parallel_{pid}_{dataset}',gpu)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int);args=parser.parse_args()
    if args.gpu is not None:worker(args.gpu);return
    if all((RUN/f'results/generation/{pid}/{ds}/metrics_0000_end.json').exists() for pid,ds in JOBS):return
    from resources import quality_gpus
    with quality_gpus(maximum=3,minimum=1) as cards:
        children=[]
        for gpu in cards:
            handle=(RUN/f'logs/parallel_generation_gpu{gpu}_{time.time_ns()}.log').open('w')
            child=subprocess.Popen([sys.executable,__file__,'--gpu',str(gpu)],stdout=handle,stderr=subprocess.STDOUT)
            children.append((child,handle))
        while any(child.poll() is None for child,_ in children):
            complete=sum((RUN/f'results/generation/{pid}/{ds}/metrics_0000_end.json').exists() for pid,ds in JOBS)
            write_json(RUN/'results/parallel_generation_progress.json',{'status':'running','gpus':cards,'complete':complete,'total':6})
            time.sleep(10)
        codes=[child.returncode for child,_ in children]
        for _,handle in children:handle.close()
        assert all(code==0 for code in codes),codes
        assert all((RUN/f'results/generation/{pid}/{ds}/metrics_0000_end.json').exists() for pid,ds in JOBS)
        write_json(RUN/'results/parallel_generation_progress.json',{'status':'complete','gpus':cards,'complete':6,'total':6,'worker_exit_codes':codes})


if __name__=='__main__':main()
