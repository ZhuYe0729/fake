"""Complete missing timings from the prospectively frozen five-budget grid."""
import csv
import fcntl
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
NIGHT=RUN.parents[1]/'night151'


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def state(stage,status='running',**details):
    write(RUN/'results/worker.json',{'status':status,'stage':stage,**details})


def invoke(argv,label):
    command=['numactl','--all','--physcpubind=0-7','--membind=0',*map(str,argv)]
    log=RUN/f'logs/{label}_{time.time_ns()}.log'
    write(log.with_suffix('.command.json'),{'command':command,'gpu':0,
        'local_code_sha256':{p.name:sha(p) for p in (RUN/'code').glob('*.py')}})
    start=time.perf_counter()
    with log.open('w') as handle:
        result=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=handle,stderr=subprocess.STDOUT)
    write(log.with_suffix('.completion.json'),{'returncode':result.returncode,'seconds':time.perf_counter()-start})
    assert result.returncode==0,str(log)


def checkpoint(rid,pid):
    source=RUN.parent/rid
    spec=read(source/'configs/model.json')
    if pid=='uniform_dense_bf16':
        model=Path(spec['temp'])/'checkpoints'/pid
        assert (model/'model.safetensors').exists()
        return model
    model=RUN/'data/checkpoints'/rid/pid
    if (model/'model.safetensors').exists():return model
    if model.exists():model.rename(model.with_name(pid+'_failed_'+str(time.time_ns())))
    assert shutil.disk_usage(RUN).free>100*1024**3
    argv=[sys.executable,source/'code/export_model.py','--model-path',spec['model_path'],
        '--policy-json',RUN/f'reference/{rid}/{pid}.json','--output-dir',model,
        '--cutlass-wrapper-path',RUN.parents[3]/'fake/kernels/cutlass/cutlass_wrapper']
    content=(RUN/f'reference/{rid}/{pid}.json').read_text()
    for method in ['sparse_bf16','sparse_nvfp4']:
        if method in content:argv+=['--canonical-'+method.replace('_','-')+'-state',Path(spec['canonical_root'])/method/'model.pt']
    with (NIGHT/'export.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        invoke(argv,f'export_{rid}_{pid}')
    return model


def run():
    design=read(RUN/'configs/design.json')
    protected={}
    for rid,model in design['models'].items():
        source=RUN.parent/rid
        while not (source/'results/final_audit.json').exists():
            state('original_models_core_tables','waiting',source_run=rid);time.sleep(10)
        assert read(source/'results/final_audit.json')['verified']
        protected[rid]=sha(source/'results/manifest.json')
        assert sha(RUN.parent/f'021/results/selected_{rid}.json')==model['selection_sha256']
        for pid,digest in model['policy_sha256'].items():
            assert sha(RUN/f'reference/{rid}/{pid}.json')==digest
            assert sha(source/f'configs/policies/{pid}.json')==digest
        # Preserve the exact engine sources used by this derived campaign.
        shutil.copytree(source/'code',RUN/'reference'/f'code_{rid}',dirs_exist_ok=True)
    write(RUN/'results/protected_sources.json',protected)
    sys.path.insert(0,str(RUN.parent/'017/code'))
    from resources import speed_gpu
    from telemetry import nvml
    handle=nvml.nvmlDeviceGetHandleByIndex(0)
    state('budget_speed','waiting')
    with speed_gpu():
        for rid,spec in design['models'].items():
            for number,order in enumerate(spec['rounds']):
                for index,label in enumerate(order):
                    target=RUN/f'models/{rid}/results/measurements/round{number}_{label}.json'
                    if target.exists():continue
                    while nvml.nvmlDeviceGetMemoryInfo(handle).used>512*1024**2:time.sleep(10)
                    pid='uniform_dense_bf16' if label.endswith('_bracket_end') else label
                    state('budget_speed',source_run=rid,round=number,index=index,policy=pid)
                    model=checkpoint(rid,pid)
                    outputs=[1,64] if pid=='uniform_dense_bf16' else spec['policies'][pid]
                    invoke([sys.executable,RUN/'code/speed_worker.py','--source-run',rid,
                        '--checkpoint',model,'--policy',pid,'--label',f'round{number}_{label}',
                        '--outputs',*outputs],f'speed_{rid}_round{number}_{label}')
    rows=[]
    for rid,spec in design['models'].items():
        for pid,outputs in spec['policies'].items():
            for output in outputs:
                ratios=[]
                for number in range(3):
                    folder=RUN/f'models/{rid}/results/measurements'
                    data=read(folder/f'round{number}_{pid}.json')
                    case=next(c for c in data['cases'] if c['output']==output)
                    samples=[s for s in case['samples'] if not s['warmup']]
                    anchors=[]
                    for label in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']:
                        baseline=read(folder/f'round{number}_{label}.json')
                        anchors += [s for c in baseline['cases'] if c['output']==output for s in c['samples'] if not s['warmup']]
                    assert len(samples)==8 and len(anchors)==16
                    ratios.append(statistics.median(s['e2e_ms'] for s in anchors)/statistics.median(s['e2e_ms'] for s in samples))
                rows.append({'source_run':rid,'policy':pid,'scenario':'prefill_only' if output==1 else 'prefill_decode',
                    'e2e_speedup':statistics.median(ratios),'min_speedup':min(ratios),'max_speedup':max(ratios),
                    'normalization':'run022 own BF16 start/end brackets, GPU0/NUMA0'})
    with (RUN/'results/budget_speed.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for rid,digest in protected.items():assert sha(RUN.parent/f'{rid}/results/manifest.json')==digest
    write(RUN/'results/final_audit.json',{'complete':True,'verified':True,'new_policy_scenarios':len(rows),
        'timed_processes':24,'source_manifests_sha256':protected,'design_sha256':sha(RUN/'configs/design.json'),
        'csv_sha256':sha(RUN/'results/budget_speed.csv'),
        'scope':'Additional same-input formal timings only; quality/selection inherited unchanged from prospective run021.'})
    state('budget_speed_complete','complete',new_policy_scenarios=len(rows))


if __name__=='__main__':
    try:run()
    except Exception as error:
        state('failed','failed',error=repr(error));raise
