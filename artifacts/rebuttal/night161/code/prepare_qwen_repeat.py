"""Freeze a current-protocol repeat including all strong Qwen comparators."""
import csv
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    run=RUNS/'048';assert not run.exists()
    for name in ['code','configs/policies','data','reference','results','logs','extensions/sparse_nvfp4']:(run/name).mkdir(parents=True,exist_ok=True)
    shutil.copytree(RUNS/'040/code',run/'code',dirs_exist_ok=True)
    for name in ['generate_tasks.py','benchmark.py']:shutil.copy2(RUNS/'033/code'/name,run/'code'/name)
    shutil.copy2(RUNS/'043/code/baseline_resources.py',run/'code/resources.py')
    spec=read(RUNS/'040/configs/model.json');spec.update(run='048',temp=str((run/'data/work').resolve()));write(run/'configs/model.json',spec)
    source=read(RUNS/'040/configs/design.json');rows=source['rows'];policies={}
    for row in rows:
        pid=row['policy'];shutil.copy2(RUNS/f'040/configs/policies/{pid}.json',run/f'configs/policies/{pid}.json')
    pid='uniform_sq_fp8';template=read(run/'configs/policies/uniform_fp8.json');template['policy_id']=pid
    template['method_map']={k:{'prefill_method':'sq_fp8','decode_method':'sq_fp8'}for k in template['method_map']}
    assert template['method_map']==read(RUNS/'002/configs/policies/uniform_sq_fp8.json')['method_map']
    write(run/f'configs/policies/{pid}.json',template)
    rows.insert(4,{'method':'SmoothQuant FP8 alpha=.5','policy':pid,'source_run':'002','budget':None})
    table=ROOT.parent/'responses/reviewer2/qwen3_table2_run016';legacy=list(csv.DictReader((table/'main_table.csv').open()));assert len(legacy)==24
    previous={'uniform_dense_bf16':('032','uniform_dense_bf16'),'uniform_int8':('028','uniform_int8'),
        'uniform_sq_int8':('027/full','uniform_sq_int8'),'uniform_fp8':('002','uniform_fp8'),
        'uniform_sq_fp8':('002','uniform_sq_fp8'),'fixed_sq_int8_w4a16':('027/full','sq_int8_w4a16'),
        'w1_e0.01':('032','w1_e0.01'),'w32_e0.03':('033','w32_e0.03')}
    oldrow={'uniform_dense_bf16':0,'uniform_int8':6,'uniform_sq_int8':7,'uniform_fp8':8,'uniform_sq_fp8':9,
        'fixed_sq_int8_w4a16':7,'hawq_signed_prefill_decode_phase_pair_r0.8':11,
        'hardware_minimum_prefill_decode_phase_pair':12,'w1_e0.01':22,'w32_e0.03':23}
    old_sources={};inputs={}
    for row in rows:
        pid=row['policy'];row['prefill_reuse_method']=legacy[oldrow[pid]]['method']
        policies[pid]=sha(run/f'configs/policies/{pid}.json')
        for task in ['cnn_dm_1000','dsum','IWSLT']:
            if pid not in previous:continue
            rid,oldpid=previous[pid];path=RUNS/rid/f'results/generation/{oldpid}/{task}/metrics_0000_end.json'
            assert path.exists();old_sources[pid,task]={'file':str(path.resolve()),'sha256':sha(path)}
    for name in ['activation_scales.pt',*[f'generation_{t}.jsonl'for t in ['cnn_dm_1000','dsum','IWSLT']]]:
        src=RUNS/('002/data/'+name if name=='activation_scales.pt'else'001/data/'+name)
        (run/'data'/name).symlink_to(src.resolve());inputs[str(src.resolve())]=sha(src)
    shutil.copy2(RUNS/'033/configs/speed_formal.json',run/'configs/speed_inputs.json')
    inputs[str((run/'configs/speed_inputs.json').resolve())]=sha(run/'configs/speed_inputs.json')
    shutil.copy2(RUNS/'040/reference/source_model_fingerprint.json',run/'reference/source_model_fingerprint.json')
    rounds=[]
    for ri in range(3):
        middle=[p for p in policies if p!='uniform_dense_bf16'];random.Random(172048+ri).shuffle(middle)
        rounds.append(['uniform_dense_bf16',*middle,'uniform_dense_bf16_bracket_end'])
    jobs=[{'policy':r['policy'],'dataset':t}for r in rows for t in ['cnn_dm_1000','dsum','IWSLT']]
    names=['common.py','eightbit.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py','generate_tasks.py','benchmark.py','export_model.py']
    write(run/'configs/design.json',{'created_at_unix':time.time(),'rows':rows,'policies':policies,'jobs':jobs,'rounds':rounds,
        'numeric_sha256':{name:sha(run/'code'/name)for name in names},'numeric_frozen_unix':time.time(),
        'previous_generation':{p:{t:v for(pid,t),v in old_sources.items()if pid==p}for p in previous},
        'legacy_table':str(table.resolve()),'legacy_table_sha256':{n:sha(table/n)for n in ['main_table.csv','README.md','provenance.json']},
        'input_sha256':inputs,'scope':'Ten frozen maps, current common Graph protocol,30 full generation jobs and33 formal processes. No refitting or filtering by results; Prefill-only MC reused independently.'})
    (run/'README.md').write_text('# Qwen十配置统一完整重复\n\n为核实002归档与028 Graph INT8分数差异，040原九map加SQ-FP8统一当前数值/Graph路径，完整1000/1500/333生成及三轮速度。旧24行保留，另建34行追加表；新行Prefill-only结果明确复用原独立配置，P+D全部新测。等待044–046完成后使用空闲GPU，047排在其后。\n\n[设计](configs/design.json) · [计划172](../../../../dev/plans/172_qwen_full_generation_repeat_plan.md)\n')
    print('Frozen048:10 maps,30 full generation jobs,33 formal processes',flush=True)


if __name__=='__main__':main()
