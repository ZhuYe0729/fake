"""Run mixed-token controls on the same card after overhead timing finishes."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import RUN,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--quality',action='store_true');args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    previous=RUN.parent/'004';assert json.loads((previous/'results/runtime_worker.json').read_text())['status']=='complete'
    design=json.loads((RUN/'configs/design.json').read_text());state=RUN/f'results/worker_{"quality" if args.quality else "speed"}.json'
    write_json(state,{'status':'running','gpu':6,'pid':os.getpid()})
    try:
        for policy in design['policies']:
            if args.quality and policy not in ['uniform_dense_bf16','phase_sq_int8_w4a16_ours','transfer_daso8_prefill_decode_01']:continue
            model=Path('/dev/shm/daso_rebuttal_004_20261025/checkpoints')/policy
            if not (model/'model.safetensors').exists():model=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/policy
            assert (model/'model.safetensors').exists(),model
            variants=[(1024,False),(4096,False)]
            if not args.quality and policy=='uniform_dense_bf16':variants=[(1024,True),(4096,True)]+variants
            if args.quality:variants=[(1024,True),(1024,False),(4096,False)]
            for budget,flag in variants:
                name=f'{policy}_{budget}_{flag}_{args.quality}';marker=RUN/f'results/complete/{name}.json'
                if marker.exists():continue
                command=[sys.executable,RUN/'code/benchmark.py','--checkpoint',str(model),'--policy',policy,'--budget',str(budget)]
                if args.quality:
                    command+=['--quality']
                    if flag:command+=['--no-chunk']
                elif flag:command+=['--noop']
                stamp=str(time.time_ns());log=RUN/f'logs/{name}_{stamp}.log';files=sorted((RUN/'code').glob('*.py'))
                hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
                version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16];snapshot=RUN/'code_versions'/version
                if not snapshot.exists():
                    snapshot.mkdir(parents=True)
                    for f in files:shutil.copy2(f,snapshot/f.name)
                write_json(log.with_suffix('.command.json'),{'command':command,'code_version':version,'code_hashes':hashes,'gpu':6})
                start=time.perf_counter()
                with log.open('w') as handle:result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
                write_json(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
                if result.returncode:raise RuntimeError(f'exit {result.returncode}: {log}')
                write_json(marker,{'complete':True});print(name,'complete',flush=True)
        write_json(state,{'status':'complete','gpu':6,'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
