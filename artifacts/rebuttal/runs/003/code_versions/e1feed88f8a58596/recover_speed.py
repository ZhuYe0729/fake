"""Rerun every final speed on idle GPU0; preserve GPU3 partial evidence."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from common import MODEL, RUN, WRAPPER, write_json
from gpu_handoff import handoff
from orchestrate import invoke

TEMP=Path('/dev/shm/daso_rebuttal_003_gpu0_20261025')
OLD=Path('/dev/shm/daso_rebuttal_003_20261025')


def cleanup():
    for directory in (TEMP/'checkpoints').glob('*'):
        policy=directory.name;binary=directory/'model.safetensors'
        if not binary.exists() or not (RUN/f'results/speed_final_complete/{policy}.json').exists():continue
        if not (RUN/f'results/quality_complete/{policy}.json').exists():continue
        size=binary.stat().st_size;binary.unlink()
        write_json(RUN/f'results/exports_final/{policy}/temporary_binary_cleanup.json',{
            'path':str(binary),'bytes':size,'reason':'all final GPU0 speed and frozen quality complete',
            'rebuild':'archived policy + original exporter/model/canonical sparse states/activation scales',
            'preserved':'all code, logs, metadata, results, CSV and documents'})


def checkpoint(policy):
    if policy.startswith('uniform_'):
        cached=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/policy
        if (cached/'model.safetensors').exists():return cached
    target=TEMP/'checkpoints'/policy
    if (target/'model.safetensors').exists() and (target/'phase_hetero_export_provenance.json').exists():return target
    while shutil.disk_usage('/dev/shm').free<32*1024**3:
        cleanup();time.sleep(2)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():target.rename(target.with_name(target.name+'_previous_'+str(time.time_ns())))
    previous=OLD/'checkpoints'/policy;linked=False
    if (previous/'model.safetensors').exists():
        try:shutil.copytree(previous,target,copy_function=os.link,symlinks=True);linked=True
        except FileNotFoundError:
            target.rename(target.with_name(target.name+'_partial_'+str(time.time_ns())))
    if not linked:
        command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',
            RUN/f'configs/policies/{policy}.json','--output-dir',target,'--cutlass-wrapper-path',WRAPPER]
        content=(RUN/f'configs/policies/{policy}.json').read_text()
        for method in ['sparse_bf16','sparse_nvfp4']:
            if method in content:command+=['--canonical-'+method.replace('_','-')+'-state',
                Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt']
        invoke(command,policy+'_export_gpu0')
    evidence=RUN/f'results/exports_final/{policy}';evidence.mkdir(parents=True,exist_ok=True)
    for source in target.glob('phase_hetero*.json'):shutil.copy2(source,evidence/source.name)
    shutil.copy2(target/'config.json',evidence/'config.json')
    write_json(evidence/'source.json',{'checkpoint':str(target),'reused_via_hardlinks':linked,
        'previous_checkpoint':str(previous) if linked else None,'rule':'same frozen policy/exporter/weights; no refit'})
    return target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cleanup-only',action='store_true');args=parser.parse_args()
    if args.cleanup_only:cleanup();return
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    os.environ['DASO_GPU_HANDOFF_DIR']=str(RUN/'results/gpu0_reservation')
    state=RUN/'results/worker_speed.json'
    write_json(state,{'status':'running','pid':os.getpid(),'gpu':'0','scope':'ALL final speeds; GPU3 partial retained'})
    try:
        frozen=json.loads((RUN/'configs/evaluation_points.json').read_text())
        # Finish previously unmeasured policies first so quality workers keep busy.
        policies=sorted(frozen['policies'],key=lambda p:(RUN/f'results/speed_complete/{p}.json').exists())
        for policy in policies:
            complete=RUN/f'results/speed_final_complete/{policy}.json'
            if complete.exists():continue
            handoff('hold');cleanup();model=checkpoint(policy)
            workloads=frozen['policies'][policy]['workloads']
            print('GPU0 speed started',policy,len(workloads),flush=True)
            invoke([sys.executable,RUN/'code/measure_matrix.py','--checkpoint',model,'--policy',policy,
                '--mode','speed','--workloads',','.join(workloads)],policy+'_speed_gpu0')
            handoff('hold')
            write_json(complete,{'complete':True,'gpu':0,'policy':policy,'checkpoint':str(model)})
            write_json(RUN/f'results/speed_complete/{policy}.json',{'complete':True,'gpu':0,'policy':policy,'checkpoint':str(model)})
            cleanup();print('GPU0 speed complete',policy,flush=True)
        write_json(RUN/'results/all_speed_complete.json',{'complete':True,'gpu':0})
        handoff('stop');write_json(state,{'status':'complete','pid':os.getpid(),'gpu':'0'})
    except Exception as error:
        write_json(state,{'status':'failed','pid':os.getpid(),'gpu':'0','error':repr(error)});raise


if __name__=='__main__':main()
