"""CPU preflight: real data exclusion and selection without confirmation files."""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    os.sched_setaffinity(0,set(range(48,56)))
    from prepare_fresh_quality import hashes
    proof=[]
    for rid in ['044','045','046']:
        run=RUNS/rid;design=read(run/'configs/design.json');purge=read(run/'reference/data_purge.json')
        forbidden=[]
        for name,digest in purge['sources_sha256'].items():
            path=Path(name);assert sha(path)==digest
            forbidden.append(hashes(torch.load(path,weights_only=True).numpy()).ravel())
        forbidden=np.unique(np.concatenate(forbidden));starts=[];fold_hashes=[]
        for split,data in design['datasets'].items():
            path=Path(data['file']);source=Path(data['source_file'])
            assert sha(path)==data['sha256']and sha(source)==data['source_sha256']
            blocks=torch.load(path,weights_only=True);tokens=torch.load(source,weights_only=True)
            assert tuple(blocks.shape)==(64,2112)
            assert not np.isin(hashes(blocks.numpy()),forbidden).any()
            fold_hashes.append(hashes(blocks.numpy()).ravel())
            for block,start in zip(blocks,data['starts'],strict=True):
                assert start%2112==0 and torch.equal(block,tokens[start:start+2112])
            starts+=data['starts']
        assert len(starts)==len(set(starts))==128
        assert np.intersect1d(*fold_hashes).size==0
        result={'verified':True,'validation_windows':64,'confirmation_windows':64,'shared_calibration_or_old_eval16grams':0,
            'nonoverlapping_selected_starts':128,'design_sha256':sha(run/'configs/design.json')}
        write(run/'results/protocol_cpu_checks.json',result);proof.append({'run':rid,**result})
    sys.path.insert(0,str(RUNS/'044/code'))
    import fresh_report
    fixture=ROOT/f'protocol_check/selection_fixture_{time.time_ns()}'
    policies={'uniform_dense_bf16':(0.,30.),'uniform_int8':(.012,20.),'uniform_sq_int8':(.028,10.),'uniform_fp8':(.009,25.)}
    write(fixture/'configs/design.json',{'datasets':{'validation':{'sha256':'fixture'}},'validation_sizes':[32,64],
        'budgets':[.01,.02,.04,.07],'policies':{p:{'linear_ms':c}for p,(_,c)in policies.items()},'selection':'fixture:minimum latency passing empirical guard'})
    for pid,(delta,_)in policies.items():
        write(fixture/f'results/validation/{pid}/prefill_decode.json',{'data_sha256':'fixture','quality_score_batch':8,
            'blocks':[{'token_count':64,'avg_nll':1.+delta}for _ in range(64)]})
    fresh_report.RUN=fixture;fresh_report.select()
    chosen=read(fixture/'configs/fresh_selection.json')['selected']
    expected=['uniform_dense_bf16','uniform_int8','uniform_sq_int8','uniform_sq_int8']*2
    assert [p['policy']for p in chosen]==expected
    assert not(fixture/'results/confirmation').exists()
    write(ROOT/'protocol_check/VERIFIED.json',{'verified':True,'data_checks':proof,'selection_fixture':str(fixture),
        'selection_without_confirmation_files':True,'expected_policies':expected,
        'code_sha256':sha(Path(__file__)),'prepare_sha256':sha(ROOT/'code/prepare_fresh_quality.py')})
    print('Verified all3 purged data partitions and8 correct selections without confirmation access',flush=True)


if __name__=='__main__':main()
