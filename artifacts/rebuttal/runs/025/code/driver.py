"""Use one idle card for metadata-only probes after the source table seals."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
SOURCE=RUN.parent/'017'


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    while not (SOURCE/'results/final_audit.json').exists():
        write(RUN/'results/worker.json',{'status':'waiting','stage':'source_core_audit'});time.sleep(10)
    assert read(SOURCE/'results/final_audit.json')['verified']
    protected=sha(SOURCE/'results/manifest.json')
    sys.path.insert(0,str(SOURCE/'code'))
    from resources import quality_gpus
    with quality_gpus(maximum=1,minimum=1) as cards:
        gpu=cards[0]
        for pid in read(RUN/'configs/design.json')['policies']:
            target=RUN/f'results/{pid}.json'
            if target.exists():continue
            command=['numactl','--all',f'--physcpubind={gpu*8}-{gpu*8+7}',f'--membind={0 if gpu<4 else 1}',
                sys.executable,str(RUN/'code/interface_probe.py'),'--policy',pid]
            log=RUN/f'logs/{pid}_{time.time_ns()}.log'
            write(log.with_suffix('.command.json'),{'command':command,'gpu':gpu,'probe_sha256':sha(RUN/'code/interface_probe.py')})
            write(RUN/'results/worker.json',{'status':'running','stage':'interface_probe','gpu':gpu,'policy':pid})
            start=time.perf_counter()
            with log.open('w') as handle:
                result=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=handle,stderr=subprocess.STDOUT)
            write(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
            assert result.returncode==0,str(log)
    rows=[]
    for pid in read(RUN/'configs/design.json')['policies']:
        data=read(RUN/f'results/{pid}.json');assert data['passed'] and len(data['records'])==256
        for row in data['records']:
            rows.append({'policy':pid,'module':row['module'],'phase':row['phase'],'weight_method':row['weight_method'],
                **{side+'_'+key:json.dumps(value) for side in ['input','output'] for key,value in row[side].items()}})
    with (RUN/'results/interfaces.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert sha(SOURCE/'results/manifest.json')==protected
    write(RUN/'results/final_audit.json',{'complete':True,'verified':True,'records':len(rows),
        'source_manifest_sha256':protected,'csv_sha256':sha(RUN/'results/interfaces.csv'),
        'scope':'Tensor-boundary dtype/shape/stride verification only; no timing or quality measurement.'})
    write(RUN/'results/worker.json',{'status':'complete','stage':'interfaces_audited','records':len(rows)})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/'results/worker.json',{'status':'failed','error':repr(error)});raise
