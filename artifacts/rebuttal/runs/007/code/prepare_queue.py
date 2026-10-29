"""Prepare simple baselines on idle GPU5 after the untraced speed stage."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from common import RUN,write_json


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5'
    marker=RUN.parent/'004/results/runtime_worker.json'
    while True:
        value=json.loads(marker.read_text()) if marker.exists() else {}
        if value.get('status')=='failed':raise RuntimeError('inspect run004 first')
        if value.get('status')=='complete':break
        time.sleep(10)
    state=RUN/'results/preparation_worker.json';write_json(state,{'status':'running','gpu':5,'pid':os.getpid()})
    try:
        for method in ['magnitude','wanda']:
            if (RUN/f'results/canonical_{method}.json').exists():continue
            while shutil.disk_usage('/dev/shm').free<24*1024**3:time.sleep(10)
            command=[sys.executable,str(RUN/'code/prepare_pruning.py'),'--method',method]
            log=RUN/f'logs/prepare_{method}_{time.time_ns()}.log';files=sorted((RUN/'code').glob('*.py'))
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
            snapshot=RUN/'code_versions'/version;snapshot.mkdir(parents=True,exist_ok=True)
            for p in files:
                if not (snapshot/p.name).exists():shutil.copy2(p,snapshot/p.name)
            write_json(log.with_suffix('.command.json'),{'command':command,'code_version':version,'code_hashes':hashes,'gpu':5})
            start=time.perf_counter()
            with log.open('w') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
            write_json(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
            if result.returncode:raise RuntimeError(f'exit {result.returncode}: {log}')
            print(method,'complete',flush=True)
        write_json(state,{'status':'complete','gpu':5,'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
