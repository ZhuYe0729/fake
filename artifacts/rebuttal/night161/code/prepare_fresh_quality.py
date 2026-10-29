"""Freeze fixed incumbents and purged train-corpus windows before GPU outputs."""
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs';K=16;L=2112
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def hashes(values):
    n=values.shape[-1]-K+1;out=np.zeros((*values.shape[:-1],n),dtype=np.uint64)
    for j in range(K):out+=values[...,j:j+n].astype(np.uint64)*np.uint64(pow(1000003,j,2**64))
    return out


def main():
    os.sched_setaffinity(0,set(range(48,56)))
    for rid,current,data_source,point_sources in [('044','033','001',['032','033']),
        ('045','034','017',['034','036']),('046','035','018',['035'])]:
        run=RUNS/rid;assert not run.exists()
        for name in ['code','configs/policies','data','reference','results','logs','extensions/sparse_nvfp4']:(run/name).mkdir(parents=True,exist_ok=True)
        shutil.copytree(RUNS/current/'code',run/'code',dirs_exist_ok=True)
        if rid=='044':
            shutil.copy2(RUNS/'034/code/launch.py',run/'code/launch.py')
            spec={'run':rid,'model_id':'qwen3_8b','model_path':'/root/wja/data/models/Qwen/Qwen3-8B',
                'temp':str((run/'data/work').resolve()),'evaluation_kv_gib':6,
                'canonical_root':str((ROOT.parent/'night151/cache/daso_rebuttal_001_20261025/canonical').resolve())}
        else:
            spec=read(RUNS/current/'configs/model.json');spec.update(run=rid,temp=str((run/'data/work').resolve()))
        write(run/'configs/model.json',spec)
        (run/'data/activation_scales.pt').symlink_to((RUNS/current/'data/activation_scales.pt').resolve())
        profile=RUNS/current/'reference/profile.csv';shutil.copy2(profile,run/'reference/profile.csv')
        lut={(r['module_type'],r['method'],int(r['m'])):float(r['latency_ms'])for r in csv.DictReader(profile.open())}
        anchors=[(current,p)for p in ['uniform_dense_bf16','uniform_int8','uniform_sq_int8','uniform_fp8','phase_sq_int8_w4a16_ours']]
        if rid=='044':anchors=[('032','uniform_dense_bf16'),('032','uniform_int8'),('032','uniform_sq_int8'),('019','uniform_fp8'),('019','fixed_sq_int8_w4a16')]
        origins=[{'run':r,'policy':p,'label':p,'role':'reference'}for r,p in anchors]
        for source in point_sources:
            points=RUNS/source/('configs/points.json'if rid=='044'else'configs/sweep_points.json')
            origins += [{'run':source,'policy':p['policy'],'label':p['label'],'role':'frozen exploratory point',
                'q':p['budget'],'prefill_weight':p['prefill_weight']}for p in read(points)]
        policies={};seen={};protected={}
        for entry in origins:
            src=RUNS/entry['run']/f'configs/policies/{entry["policy"]}.json';value=read(src);mapping=value['method_map']
            signature=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
            protected[str(src.resolve())]=sha(src)
            if signature in seen:policies[seen[signature]]['origins'].append(entry);continue
            pid=entry['policy'];assert pid not in policies;seen[signature]=pid
            dest=run/f'configs/policies/{pid}.json';shutil.copy2(src,dest)
            cost=sum(lut[name.rsplit('.',1)[-1],pair['prefill_method'],16384]+63*lut[name.rsplit('.',1)[-1],pair['decode_method'],8]
                for name,pair in mapping.items())
            policies[pid]={'origins':[entry],'map_sha256':signature,'policy_sha256':sha(dest),'linear_ms':cost}
        data=RUNS/data_source/'data';train_path=data/'wikitext_train_tokens.pt';tokens=torch.load(train_path,weights_only=True)
        purge_paths=[data/'calibration_128_2112.pt',data/'calibration_train_100_2048.pt',
            data/'wikitext_validation_tokens.pt',data/'wikitext_test_tokens.pt']
        forbidden=np.unique(np.concatenate([hashes(torch.load(p,weights_only=True).numpy()).ravel()for p in purge_paths]))
        source_hashes=hashes(tokens.numpy());matches=np.isin(source_hashes,forbidden)
        eligible=[s for s in range(0,len(tokens)-L+1,L)if not matches[s:s+L-K+1].any()]
        assert len(eligible)>=128,(rid,len(eligible))
        seed=168000+int(rid);starts=eligible.copy();random.Random(seed).shuffle(starts)
        validation_starts=starts[:64]
        validation_hashes=np.unique(np.concatenate([source_hashes[s:s+L-K+1]for s in validation_starts]))
        confirmation_starts=[s for s in starts[64:]if not np.isin(source_hashes[s:s+L-K+1],validation_hashes).any()][:64]
        assert len(confirmation_starts)==64
        datasets={}
        for split,selected in [('validation',validation_starts),('confirmation',confirmation_starts)]:
            blocks=torch.stack([tokens[s:s+L]for s in selected])
            assert not np.isin(hashes(blocks.numpy()),forbidden).any()
            path=run/f'data/{split}_64_2112.pt';torch.save(blocks,path)
            datasets[split]={'file':str(path.resolve()),'blocks':64,'sha256':sha(path),'starts':selected,
                'source_split':'wikitext train','source_file':str(train_path.resolve()),'source_sha256':sha(train_path),
                'scope':'New windows purged against all experimental calibration and full original validation/test16grams; public train corpus, not external unseen corpus.'}
        assert not(set(datasets['validation']['starts'])&set(datasets['confirmation']['starts']))
        write(run/'configs/frozen_design.json',{'datasets':datasets})
        write(run/'configs/selected_policies.json',{'selected':list(policies),'scope':'Frozen finite pool, selections for confirmation written only after validation.'})
        purge={'k':K,'length':L,'seed':seed,'eligible_aligned_windows':len(eligible),'total_aligned_windows':len(tokens)//L,
            'forbidden_hash_count':len(forbidden),'matched_source_positions':int(matches.sum()),'fresh_crossfold_shared16gram_hashes':0,
            'sources_sha256':{str(p.resolve()):sha(p)for p in purge_paths},
            'rule':'Reject any window sharing16-token hash with calibration or complete old validation/test streams. Hash collisions can only reject extra windows.'}
        write(run/'reference/data_purge.json',purge)
        for source in point_sources:
            for filename in ['configs/points.json','configs/sweep_points.json']:
                path=RUNS/source/filename
                if path.exists():protected[str(path.resolve())]=sha(path)
        design={'source_current':current,'data_source':data_source,'point_sources':point_sources,'policies':policies,
            'budgets':[.01,.02,.04,.07],'validation_sizes':[32,64],'guard':'paired mean+2SE+.005, BF16=0',
            'selection':'Minimum fixed Linear LUT cost among passing frozen incumbents; tie guard,policy ID. No new full-space solve.',
            'datasets':datasets,'purge_sha256':sha(run/'reference/data_purge.json'),'protected_sha256':protected,
            'profile_sha256':sha(run/'reference/profile.csv'),'activation_scales_sha256':sha(run/'data/activation_scales.pt'),
            'scope':'Prospective window-level audit of already explored fixed policies; bounded pool; no downstream task or speed rerun.'}
        write(run/'configs/design.json',design)
        score=run/'code/score_nll.py';text=score.read_text();text=text.replace('Fixed training protocol or new nonoverlapping confirmation; no test-based selection.',
            'Fresh purged public train-corpus windows; validation selected before own confirmation, not external unseen corpus.')
        score.write_text(text)
        (run/'README.md').write_text(f'# {spec["model_id"]}固定池新窗口质量审计\n\n{len(policies)}套固定map，新validation64与confirmation64。排除旧校准和完整旧validation/test的共享16token片段。公开train语料，非外部新语料；只复核NLL预算，不新增下游任务/速度。等待043完成后启动。\n\n[固定设计](configs/design.json) · [数据排除](reference/data_purge.json) · [计划](../../../../dev/plans/168_fixed_pool_fresh_quality_audit_plan.md)\n')
        print('Frozen',rid,spec['model_id'],len(policies),'maps',len(eligible),'eligible purged windows',flush=True)


if __name__=='__main__':main()
