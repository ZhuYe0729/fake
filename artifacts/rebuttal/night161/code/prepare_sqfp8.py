"""Freeze plain and smoothed FP8 controls under original Llama task protocols."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    for rid,original,current in [('042','017','034'),('043','018','035')]:
        run=RUNS/rid;assert not run.exists()
        for name in ['code','configs/policies','data','reference','results','logs','extensions/sparse_nvfp4']:(run/name).mkdir(parents=True,exist_ok=True)
        shutil.copytree(RUNS/current/'code',run/'code',dirs_exist_ok=True)
        spec=read(RUNS/current/'configs/model.json');spec.update(run=rid,temp=str((run/'data/work').resolve()));write(run/'configs/model.json',spec)
        for p in(RUNS/original/'data').iterdir():
            if p.is_file()and p.suffix in ['.pt','.csv','.json','.jsonl']or p.name in ['likelihood_datasets','likelihood_auxiliary_datasets']:
                (run/'data'/p.name).symlink_to(p.resolve())
        for name in ['generation_protocol.json','speed_inputs.json','full_task_protocol.json']:
            shutil.copy2(RUNS/original/'configs'/name,run/'configs'/name)
        shutil.copy2(RUNS/current/'reference/source_model_fingerprint.json',run/'reference/source_model_fingerprint.json')
        frozen=read(RUNS/original/'configs/frozen_design.json');write(run/'configs/frozen_design.json',frozen)
        anchors=['uniform_dense_bf16','uniform_int8','uniform_sq_int8','uniform_fp8'];pid='uniform_sq_fp8'
        policies={};reused={};inputs={}
        for p in anchors:
            src=RUNS/original/f'configs/policies/{p}.json';shutil.copy2(src,run/f'configs/policies/{p}.json');policies[p]=sha(src)
            paths=[RUNS/original/f'results/confirmation/{p}/{s}.json'for s in ['prefill_only','prefill_decode']]
            paths += [RUNS/original/f'results/generation/{p}/{d}/metrics_0000_end.json'for d in ['cnn_dm_1000','dsum','IWSLT']]
            paths += [RUNS/original/f'results/tasks/{p}/{t}/full/result.json'for t in ['arc_challenge','arc_easy','winogrande','mmlu']]
            for path in paths:assert path.exists();reused[str(path.resolve())]=sha(path)
        template=read(run/'configs/policies/uniform_dense_bf16.json')
        template['policy_id']=pid;template['method_map']={name:{'prefill_method':'sq_fp8','decode_method':'sq_fp8'}for name in template['method_map']}
        write(run/f'configs/policies/{pid}.json',template);policies[pid]=sha(run/f'configs/policies/{pid}.json')
        write(run/'configs/selected_policies.json',{'selected':[{'policy':p}for p in policies],'scope':'Fixed baseline, no quality selection.'})
        for path in [run/'data/activation_scales.pt',run/'configs/speed_inputs.json',run/'configs/full_task_protocol.json',
            *[run/f'data/generation_{t}.jsonl'for t in ['cnn_dm_1000','dsum','IWSLT']],Path(frozen['datasets']['confirmation']['file'])]:
            inputs[str(path.resolve())]=sha(path)
        protected=[RUNS/original/'results/final_audit.json',RUNS/original/'configs/frozen_design.json']
        for name in ['common.py','eightbit.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py','export_model.py','run_workflow.py','evaluate_prefill_tasks_batch8.py','generate_tasks.py']:
            protected.append(RUNS/original/'code'/name)
            assert sha(run/'code'/name)==sha(RUNS/original/'code'/name)
        jobs=[{'label':'full_mc','stage':'mc','policy':pid},{'label':'nll','stage':'nll','policy':pid}]
        jobs += [{'label':'generation_'+t,'stage':'generation','policy':pid,'dataset':t}for t in ['cnn_dm_1000','dsum','IWSLT']]
        write(run/'configs/design.json',{'original_run':original,'source_current':current,'anchors':anchors,'policies':policies,'jobs':jobs,
            'reused_quality_sha256':reused,'input_sha256':inputs,'protected_sha256':{str(p.resolve()):sha(p)for p in protected},
            'alpha':.5,'scope':'Existing SQ-FP8 baseline path, original scales/MC/GEN protocols; no alpha tuning or ninth-action DASO refit.'})
        (run/'README.md').write_text(f'# {spec["model_id"]}：SQ-FP8完整基线\n\n固定原alpha=.5和activation scales，完整MC/GEN及两场景64窗口NLL；与原BF16/INT8/SQ-INT8/FP8同轮三轮测速。质量等037，正式速度等041。\n\n[设计](configs/design.json) · [计划](../../../../dev/plans/167_llama_smoothquant_fp8_baseline_plan.md)\n')
        print('Prepared',rid,spec['model_id'],'five quality jobs, five maps',flush=True)


if __name__=='__main__':main()
