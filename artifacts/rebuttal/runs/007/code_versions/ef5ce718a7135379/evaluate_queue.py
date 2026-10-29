"""Evaluate every simple pruning baseline on the same card and held-out data."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import MODEL,RUN,WRAPPER,write_json

TEMP=Path('/dev/shm/daso_rebuttal_007_20261025')


def invoke(command,label):
    log=RUN/f'logs/{label}_{time.time_ns()}.log';files=sorted((RUN/'code').glob('*.py'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot=RUN/'code_versions'/version;snapshot.mkdir(parents=True,exist_ok=True)
    for p in files:
        if not (snapshot/p.name).exists():shutil.copy2(p,snapshot/p.name)
    write_json(log.with_suffix('.command.json'),{'command':list(map(str,command)),'code_version':version,'code_hashes':hashes,'gpu':5})
    start=time.perf_counter()
    with log.open('w') as f:result=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
    if result.returncode:raise RuntimeError(f'exit {result.returncode}: {log}')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5'
    while True:
        file=RUN/'results/preparation_worker.json';value=json.loads(file.read_text()) if file.exists() else {}
        if value.get('status')=='failed':raise RuntimeError('pruning preparation failed')
        if value.get('status')=='complete' and (RUN.parent/'005/configs/selected_policies.json').exists():break
        time.sleep(10)
    state=RUN/'results/evaluation_worker.json';write_json(state,{'status':'running','gpu':5,'pid':os.getpid()})
    try:
        for method in ['magnitude','wanda','sparsegpt','dense_bf16']:
            done=RUN/f'results/complete/{method}.json'
            if done.exists():continue
            if method in ['sparsegpt','dense_bf16']:
                model=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/('uniform_sparse_bf16' if method=='sparsegpt' else 'uniform_dense_bf16')
            else:
                model=TEMP/'checkpoints'/method
                if not (model/'model.safetensors').exists():
                    while shutil.disk_usage('/dev/shm').free<24*1024**3:time.sleep(10)
                    invoke([sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',RUN/f'configs/policies/{method}.json',
                        '--output-dir',model,'--cutlass-wrapper-path',WRAPPER,'--canonical-sparse-bf16-state',TEMP/'canonical'/method/'model.pt'],method+'_export')
                directory=RUN/f'results/exports/{method}';directory.mkdir(parents=True,exist_ok=True)
                for p in model.glob('*.json'):shutil.copy2(p,directory/p.name)
            assert (model/'model.safetensors').exists()
            for mode in ['test_speed','test']:
                invoke([sys.executable,RUN/'code/evaluate.py','--checkpoint',model,'--policy',method,'--mode',mode,
                        '--scenarios','prefill_only,prefill_decode'],method+'_'+mode)
            write_json(done,{'complete':True,'checkpoint':str(model)})
            if method in ['magnitude','wanda']:
                binary=model/'model.safetensors';size=binary.stat().st_size;binary.unlink()
                write_json(RUN/f'results/exports/{method}/binary_cleanup.json',{'bytes':size,'path':str(binary),
                    'reason':'all planned speed and NLL complete; canonical state remains for optional complete downstream tasks',
                    'preserved':'all code/log/CSV/result/document and export metadata'})
            print(method,'evaluation complete',flush=True)
        write_json(state,{'status':'complete','gpu':5,'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
