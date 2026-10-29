"""Run copies, exports, balanced repeats, then frozen SLA validation/test."""
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
TEMP=Path('/dev/shm/daso_rebuttal_012_20261026')
MODEL=Path('/root/wja/data/models/Qwen/Qwen3-8B')
WRAPPER=RUN.parents[3]/'fake/kernels/cutlass/cutlass_wrapper'


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')


def invoke(argv,label,node):
    prefix=['numactl','--physcpubind='+('0-7' if node==0 else '32-39'),'--membind='+str(node)]
    command=prefix+list(map(str,argv));log=RUN/f'logs/{label}_{time.time_ns()}.log'
    files=sorted((RUN/'code').glob('*.py'));hashes={p.name:digest(p) for p in files};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot=RUN/'code_versions'/version;snapshot.mkdir(parents=True,exist_ok=True)
    for p in files:
        if not (snapshot/p.name).exists():shutil.copy2(p,snapshot/p.name)
    write(log.with_suffix('.command.json'),{'command':command,'code_version':version,'code_hashes':hashes,'node':node,'gpu':0})
    start=time.perf_counter()
    with log.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    write(log.with_suffix('.completion.json'),{'returncode':r.returncode,'seconds':time.perf_counter()-start})
    if r.returncode:raise RuntimeError(str(log))


def freeze_sla(design):
    path=RUN/'configs/sla_selected.json'
    if path.exists():return
    def samples(pid):
        return [s['e2e_ms'] for i in [0,1] for c in read(RUN/f'results/speed/node0/round{i}/{pid}.json')['cases']
            if c['scenario']=='prefill_decode' for s in c['samples'] if not s['warmup']]
    assert not (RUN/'results/speed/node0/round2').exists()
    pool=[]
    for source in design['sla']['original_validation']:
        values=samples(source['policy_id']);assert len(values)==16
        pool.append({'policy':source['policy_id'],'guarded_quality':source['guarded_delta_nll'],
            'median_ms':statistics.median(values),'max_ms':max(values),'original_5_percent':statistics.median(values)*1.05,
            'empirical_upper':max(values)*1.02,'samples':values})
    reference=statistics.median(samples('uniform_dense_bf16'));points=[]
    for fraction in design['sla']['fractions']:
        for rule in design['sla']['rules']:
            feasible=[r for r in pool if r[rule]<=reference*fraction]
            winner=min(feasible,key=lambda r:(r['guarded_quality'],r[rule],r['policy'])) if feasible else None
            points.append({'fraction':fraction,'rule':rule,'sla_ms':reference*fraction,'policy':winner['policy'] if winner else None,'infeasible':winner is None})
    write(path,{'points':points,'calibration_candidates':pool,'bf16_calibration_ms':reference,'frozen_before_test':True,
        'design_sha256':digest(RUN/'configs/design.json'),'time_ns':time.time_ns()})
    print('SLA selection frozen',[(p['rule'],p['fraction'],p['policy']) for p in points],flush=True)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0';design=read(RUN/'configs/design.json');state=RUN/'results/worker.json'
    write(state,{'status':'running','pid':os.getpid(),'gpu':0})
    try:
        for i,node in enumerate(design['copy_probe']['order']):
            if not (RUN/f'results/copy/{i}.json').exists():invoke([sys.executable,RUN/'code/copy_probe.py','--index',i],f'copy_{i}_node{node}',node)
        for pid in design['policies']:
            if pid=='uniform_dense_bf16':continue
            model=TEMP/'checkpoints'/pid
            if (model/'model.safetensors').exists():continue
            assert shutil.disk_usage('/dev/shm').free>24*1024**3
            command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',RUN/f'configs/policies/{pid}.json','--output-dir',model,'--cutlass-wrapper-path',WRAPPER]
            text=(RUN/f'configs/policies/{pid}.json').read_text()
            for method in ['sparse_bf16','sparse_nvfp4']:
                if method in text:command+=['--canonical-'+method.replace('_','-')+'-state',Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt']
            invoke(command,pid+'_export',0);dest=RUN/f'results/exports/{pid}';dest.mkdir(parents=True,exist_ok=True)
            for p in model.glob('*.json'):shutil.copy2(p,dest/p.name)
        for spec in design['rounds']:
            number=spec['round']
            if number==2:freeze_sla(design)
            jobs=[(0,p) for p in spec['order']]
            if number in design['remote_rounds']:
                remote=design['remote_policies'] if number==0 else list(reversed(design['remote_policies']))
                jobs += [(1,p) for p in remote]
            for node,pid in jobs:
                target=RUN/f'results/speed/node{node}/round{number}/{pid}.json'
                if target.exists():continue
                model=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_dense_bf16') if pid=='uniform_dense_bf16' else TEMP/'checkpoints'/pid
                invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',pid,'--round',number,'--node',node],f'node{node}_round{number}_{pid}',node)
                print('complete',node,number,pid,flush=True)
        write(state,{'status':'complete','gpu':0,'pid':os.getpid()})
    except Exception as e:write(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
