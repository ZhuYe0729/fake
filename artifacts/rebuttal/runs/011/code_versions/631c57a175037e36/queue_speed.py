"""Matched real timing on idle GPU0, with beginning/end BF16 anchors."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import MODEL,RUN,WRAPPER,write_json
TEMP=Path('/dev/shm/daso_rebuttal_011_20261025')


def invoke(command,label):
    log=RUN/f'logs/{label}_{time.time_ns()}.log';files=sorted((RUN/'code').glob('*.py'))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot=RUN/'code_versions'/version;snapshot.mkdir(parents=True,exist_ok=True)
    for p in files:
        if not (snapshot/p.name).exists():shutil.copy2(p,snapshot/p.name)
    write_json(log.with_suffix('.command.json'),{'command':list(map(str,command)),'code_version':version,'code_hashes':hashes,'gpu':'0'})
    start=time.perf_counter()
    with log.open('w') as f:r=subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.completion.json'),{'returncode':r.returncode,'seconds':time.perf_counter()-start})
    if r.returncode:raise RuntimeError(str(log))


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    parser=argparse.ArgumentParser();parser.add_argument('--quiet',action='store_true');args=parser.parse_args()
    pool=json.loads((RUN/'configs/candidate_pool.json').read_text());stage='quiet' if args.quiet else 'speed';state=RUN/f'results/worker_{stage}.json'
    order=pool['speed_order'];temp=TEMP
    if args.quiet:
        assert json.loads((RUN/'results/worker_speed.json').read_text())['status']=='complete'
        assert json.loads((RUN.parent/'006/results/repeat_worker.json').read_text())['status']=='complete'
        order=['uniform_dense_bf16']+list(reversed(order[1:-1]))+['uniform_dense_bf16_bracket_end'];temp=TEMP.with_name(TEMP.name+'_quiet')
        config=RUN/'configs/quiet_campaign.json'
        if not config.exists():write_json(config,{'order':order,'pool_sha256':hashlib.sha256((RUN/'configs/candidate_pool.json').read_bytes()).hexdigest(),
            'time_ns':time.time_ns(),'reason':'Initial campaign partly overlapped run006 on another GPU. All other owned GPU experiments finished; reverse-order full union is final primary result. No policy/quality/seed/protocol change, no further repeats based on results.'})
    write_json(state,{'status':'running','gpu':0,'pid':os.getpid()})
    try:
        for pid in order:
            marker=RUN/f'results/{stage}_complete/{pid}.json'
            if marker.exists():continue
            baseline=pid.startswith('uniform_dense_bf16');model=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints/uniform_dense_bf16') if baseline else temp/'checkpoints'/pid
            if not baseline:
                assert shutil.disk_usage('/dev/shm').free>24*1024**3
                command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',RUN/f'configs/policies/{pid}.json','--output-dir',model,'--cutlass-wrapper-path',WRAPPER]
                text=(RUN/f'configs/policies/{pid}.json').read_text()
                for method in ['sparse_bf16','sparse_nvfp4']:
                    if method in text:command+=['--canonical-'+method.replace('_','-')+'-state',Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt']
                invoke(command,pid+'_'+stage+'_export');metadata=RUN/f'results/exports_{stage}/{pid}';metadata.mkdir(parents=True,exist_ok=True)
                for p in model.glob('*.json'):shutil.copy2(p,metadata/p.name)
            scenarios=pool['policies']['uniform_dense_bf16' if baseline else pid]['scenarios']
            command=[sys.executable,RUN/'code/evaluate.py','--checkpoint',model,'--policy',pid,'--mode','calibration_speed','--scenarios',','.join(scenarios)]
            if args.quiet:command+=['--quiet']
            invoke(command,pid+'_'+stage)
            write_json(marker,{'complete':True,'checkpoint':str(model)})
            if not baseline:
                binary=model/'model.safetensors';size=binary.stat().st_size;binary.unlink()
                write_json(RUN/f'results/exports_{stage}/{pid}/binary_cleanup.json',{'path':str(binary),'bytes':size,'reason':'all frozen speed jobs complete; inherited quality already archived; all source/log/CSV/results/documents retained'})
            print(pid,'complete',flush=True)
        write_json(state,{'status':'complete','gpu':0,'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
