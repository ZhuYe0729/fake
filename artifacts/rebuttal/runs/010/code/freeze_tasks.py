"""Preregister representative downstream tasks; dedup only identical weights."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old=RUN.parent/'002';source=RUN.parent/'003'
    for d in ['configs/policies','data','reference','results','logs','code/compat']:(RUN/d).mkdir(parents=True,exist_ok=True)
    for name in ['common.py','eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py']:shutil.copy2(source/'code'/name,RUN/'code'/name)
    for name in ['run_workflow.py','generate_tasks.py']:shutil.copy2(old/'code'/name,RUN/'code'/name)
    shutil.copytree(source/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    for name in ['likelihood_datasets','likelihood_auxiliary_datasets','activation_scales.pt']:
        (RUN/'data'/name).symlink_to(old/'data'/name)
    for task in ['cnn_dm_1000','dsum','IWSLT']:(RUN/f'data/generation_{task}.jsonl').symlink_to(old/f'data/generation_{task}.jsonl')
    rules={'quality_budgets':[.01,.04],'selection_family':'DASO plus validation',
        'prefill_tasks':['wikitext','winogrande','arc_easy','arc_challenge','mmlu'],
        'decode_tasks':['cnn_dm_1000','dsum','IWSLT'],'simple_pruning':['magnitude','wanda'],
        'inheritance':'Only exact method-map AND original unchanged weight/canonical/scales provenance can reuse run002. New magnitude/Wanda weights always require new evaluation despite the same sparse method-map.',
        'protocol':'Full original task counts, prompts, fewshot=0, seeds, generation limits and scoring from run002. No selection using final test scores.',
        'registered_time_ns':time.time_ns()}
    (RUN/'configs/task_rules.json').write_text(json.dumps(rules,indent=2)+'\n')
    while not (RUN.parent/'005/configs/selected_policies.json').exists():time.sleep(10)
    selection=RUN.parent/'005/configs/selected_policies.json';selected=json.loads(selection.read_text());jobs=[];policies={};inherited=[];aliases=[]
    old_maps={}
    for path in (old/'configs/policies').glob('*.json'):
        policy=json.loads(path.read_text())
        if 'method_map' in policy:old_maps.setdefault(hashlib.sha256(json.dumps(policy['method_map'],sort_keys=True).encode()).hexdigest(),[]).append(path.stem)

    def add(pid,policy,kind,origin):
        target=RUN/f'configs/policies/{pid}.json'
        if not target.exists():target.write_text(json.dumps({**policy,'policy_id':pid},indent=2)+'\n')
        policies[pid]={'origin':origin,'policy_sha256':digest(target)}
        candidates=old_maps.get(hashlib.sha256(json.dumps(policy['method_map'],sort_keys=True).encode()).hexdigest(),[]) if origin=='original canonical states' else []
        tasks=rules['prefill_tasks'] if kind=='tasks' else rules['decode_tasks']
        for previous in candidates:
            paths=[old/f'results/tasks/{previous}/{t}/full/result.json' if kind=='tasks' else old/f'results/generation/{previous}/{t}/metrics_0000_end.json' for t in tasks]
            if not all(p.exists() for p in paths):continue
            for task,source in zip(tasks,paths,strict=True):
                dest=(RUN/f'results/tasks/{pid}/{task}/full/result.json' if kind=='tasks' else RUN/f'results/generation/{pid}/{task}/metrics_0000_end.json')
                dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(os.path.relpath(source.resolve(),dest.parent))
                inherited.append({'policy':pid,'kind':kind,'task':task,'source':str(source.resolve()),'sha256':digest(source),
                    'target':str(dest.relative_to(RUN)),'reason':'same method-map, same unchanged original canonical states/scales and runtime methods'})
            return
        job={'id':pid+'_'+kind,'policy_id':pid,'kind':kind,'origin':origin}
        if job not in jobs:jobs.append(job)

    for point in selected['points']:
        if point['family']!=rules['selection_family'] or point.get('quality_budget') not in rules['quality_budgets']:continue
        original_pid=point['policy_id'];policy=json.loads((RUN.parent/f'005/configs/policies/{original_pid}.json').read_text())
        signature=hashlib.sha256(json.dumps(policy['method_map'],sort_keys=True).encode()).hexdigest()[:12]
        pid='verified_'+signature;kind='tasks' if point['scenario']=='prefill_only' else 'generation'
        aliases.append({'label':point['label'],'policy_id':pid,'kind':kind,'source_policy':original_pid})
        if not any(j['policy_id']==pid and j['kind']==kind for j in jobs) and not any(i['policy']==pid and i['kind']==kind for i in inherited):add(pid,policy,kind,'original canonical states')
    for method in rules['simple_pruning']:
        policy=json.loads((RUN.parent/f'007/configs/policies/{method}.json').read_text());add('pruning_'+method,policy,'tasks',method+' canonical state from run007')
    for label,previous in [('reference_bf16','uniform_dense_bf16'),('reference_sparsegpt','uniform_sparse_bf16')]:
        policy=json.loads((old/f'configs/policies/{previous}.json').read_text());add(label,policy,'tasks','original canonical states')
    (RUN/'configs/task_catalog.json').write_text(json.dumps({'jobs':jobs,'policies':policies,'aliases':aliases,
        'source_selection_sha256':digest(selection),'rules_sha256':digest(RUN/'configs/task_rules.json'),
        'selection_precedes_full_tasks':True},indent=2)+'\n')
    (RUN/'results/inherited_tasks.json').write_text(json.dumps({'records':inherited},indent=2)+'\n')
    print(json.dumps({'new_full_task_jobs':len(jobs),'inherited_task_results':len(inherited),'aliases':aliases}))


if __name__=='__main__':main()
