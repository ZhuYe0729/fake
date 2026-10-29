"""Validation-only selection followed by fresh, immutable confirmation."""
import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from launch import RUN,TEMP,checkpoint,invoke,read,write


def sha(path):
    with path.open('rb')as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def state(stage,**details):write(RUN/'results/worker.json',{'status':'running','stage':stage,**details})


def worker(split):
    from baseline_resources import lease
    design=read(RUN/'configs/design.json')
    policies=list(design['policies'])if split=='validation'else read(RUN/'configs/fresh_selection.json')['confirmation_policies']
    for pid in policies:
        marker=RUN/f'results/completed/{split}/{pid}.json';lock=RUN/f'results/locks/{split}_{pid}.lock'
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a')as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if marker.exists():continue
            with lease()as gpu:
                model=checkpoint(pid,gpu);binary=model/'model.safetensors';digest=sha(binary)
                write(RUN/f'results/representations/{split}/{pid}.json',{'sha256':digest,'bytes':binary.stat().st_size,
                    'policy_sha256':sha(RUN/f'configs/policies/{pid}.json'),'source_checkpoint':str(model)})
                if split=='confirmation':assert digest==read(RUN/f'results/representations/validation/{pid}.json')['sha256'],pid
                invoke([sys.executable,RUN/'code/score_nll.py','--checkpoint',model,'--policy',pid,'--split',split,
                    '--scenarios','prefill_decode'],split+'_'+pid,gpu)
                write(marker,{'complete':True,'gpu':gpu,'policy':pid,'representation_sha256':digest})
                write(RUN/f'results/cleanup/{split}_{pid}.json',{'path':str(binary),'bytes':binary.stat().st_size,
                    'reason':'Own reproducible export; numeric representation hash preserved for exact reexport check; all other artifacts retained.'})
                binary.unlink()


def queue(split):
    policies=list(read(RUN/'configs/design.json')['policies'])if split=='validation'else read(RUN/'configs/fresh_selection.json')['confirmation_policies']
    children=[]
    for i in range(3):
        log=(RUN/f'logs/{split}_worker{i}_{time.time_ns()}.log').open('w')
        children.append((subprocess.Popen([sys.executable,__file__,'--worker',split],stdout=log,stderr=subprocess.STDOUT),log))
    while any(p.poll()is None for p,_ in children):
        state(split,complete=len(list((RUN/f'results/completed/{split}').glob('*.json'))),total=len(policies));time.sleep(10)
    assert all(p.returncode==0 for p,_ in children)
    for _,log in children:log.close()
    assert len(list((RUN/f'results/completed/{split}').glob('*.json')))==len(policies)
    write(RUN/f'results/{split}_complete.json',{'complete':True,'policies':len(policies)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['validation','confirmation']);args=parser.parse_args()
    if args.worker:return worker(args.worker)
    with(RUN/'results/driver.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not(RUN.parent/'043/results/final_audit.json').exists():state('wait043');time.sleep(10)
        assert read(RUN.parent/'043/results/final_audit.json')['verified']
        source='040'if RUN.name=='044'else read(RUN/'configs/design.json')['source_current']
        origin=RUN.parent/source/'reference/source_model_fingerprint.json';fingerprint=RUN/'reference/source_model_fingerprint.json'
        if fingerprint.exists():assert fingerprint.read_bytes()==origin.read_bytes()
        else:shutil.copy2(origin,fingerprint)
        parent_audits={}
        for rid in ['032','033','034','035','036','037','038','039','040','041','042','043']:
            path=RUN.parent/rid/'results'/('sweep_final_audit.json'if rid in ['034','035','036','038','039']else'final_audit.json')
            parent_audits[str(path)]=sha(path)
        frozen_audits=RUN/'reference/parent_audits.json'
        if frozen_audits.exists():assert read(frozen_audits)==parent_audits
        else:write(frozen_audits,parent_audits)
        if not(RUN/'results/validation_complete.json').exists():queue('validation')
        from fresh_report import select
        selection=RUN/'configs/fresh_selection.json'
        if not selection.exists():
            assert not list((RUN/'results/confirmation').glob('*/*.json'))
            select()
        frozen=sha(selection);stamp=RUN/'reference/selection_frozen.json'
        if stamp.exists():assert read(stamp)['sha256']==frozen
        else:
            assert not list((RUN/'results/confirmation').glob('*/*.json'))
            write(stamp,{'sha256':frozen,'written_at_unix':time.time(),'confirmation_read':False})
        if not(RUN/'results/confirmation_complete.json').exists():queue('confirmation')
        assert sha(selection)==frozen
        state('report')
        subprocess.run([sys.executable,RUN/'code/fresh_report.py'],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        write(RUN/'results/worker.json',{'status':'complete','stage':'audited'})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/f'results/failure_{os.getpid()}_{time.time_ns()}.json',{'error':repr(error)});raise
