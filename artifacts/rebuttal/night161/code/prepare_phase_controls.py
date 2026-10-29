"""Prepare two fixed phase controls, not new optimizer proposals."""
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
    for rid,current,original in [('038','034','017'),('039','035','018')]:
        run=RUNS/rid;assert not run.exists()
        for directory in ['code','configs','data','results','logs','reference','extensions/sparse_nvfp4']:(run/directory).mkdir(parents=True,exist_ok=True)
        shutil.copytree(RUNS/current/'code',run/'code',dirs_exist_ok=True)
        spec=read(RUNS/current/'configs/model.json');spec.update(run=rid,temp=str((run/'data/work').resolve()))
        write(run/'configs/model.json',spec)
        for path in(RUNS/original/'data').iterdir():
            if path.is_file()and path.suffix in ['.pt','.json','.jsonl','.csv']:(run/'data'/path.name).symlink_to(path.resolve())
        for name in ['generation_protocol.json','speed_inputs.json']:
            shutil.copy2(RUNS/current/'configs'/name,run/'configs'/name)
        shutil.copy2(RUNS/current/'reference/source_model_fingerprint.json',run/'reference/source_model_fingerprint.json')
        if original=='018':shutil.copy2(RUNS/current/'results/source_model_complete.json',run/'results/source_model_complete.json')
        baseline=['uniform_dense_bf16','uniform_int8','uniform_sq_int8','uniform_fp8','phase_sq_int8_w4a16_ours']
        controls=['phase_int8_w4a16_ours','phase_fp8_w4a16_ours']
        registry={};reused_nll=[];new_generation=[];reused_generation=[]
        previous_jobs=read(RUNS/current/'configs/sweep_generation_jobs.json')
        for pid in baseline+controls:
            source=RUNS/original/f'configs/policies/{pid}.json';dest=run/f'configs/policies/{pid}.json'
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            registry[pid]={'role':'fixed phase control'if pid in controls else'reference','policy_sha256':sha(dest)}
            if pid in baseline:
                path=RUNS/current/f'results/confirmation/{pid}/prefill_decode.json'
                target=run/f'results/confirmation/{pid}/prefill_decode.json';target.parent.mkdir(parents=True,exist_ok=True)
                target.symlink_to(path.resolve());reused_nll.append({'policy':pid,'source':str(path.resolve()),'sha256':sha(path)})
            for task in ['cnn_dm_1000','dsum','IWSLT']:
                if pid in controls:new_generation.append({'policy':pid,'dataset':task});continue
                source=next(x for x in previous_jobs['reused']if(x['policy'],x['dataset'])==(pid,task))
                assert sha(Path(source['source']))==source['sha256'];reused_generation.append(source)
        write(run/'configs/policy_registry.json',registry);write(run/'configs/sweep_points.json',[])
        write(run/'configs/selected_policies.json',{'selected':[{'policy':pid}for pid in registry],
            'scope':'All fixed controls, not validation or optimizer selection.'})
        write(run/'configs/sweep_evaluation_jobs.json',{'jobs':[{'policy':p}for p in controls],'reused':reused_nll})
        write(run/'configs/sweep_generation_jobs.json',{'jobs':new_generation,'reused':reused_generation})
        old=read(RUNS/current/'configs/frozen_design.json');write(run/'configs/frozen_design.json',old)
        inputs=[run/'configs/model.json',run/'configs/generation_protocol.json',run/'configs/speed_inputs.json',
            run/'reference/source_model_fingerprint.json',RUNS/current/'configs/sweep_design.json',
            *[RUNS/current/'code'/f for f in ['common.py','eightbit.py','export_model.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py','generate_tasks.py']],
            *[run/f'data/generation_{d}.jsonl'for d in ['cnn_dm_1000','dsum','IWSLT']]]
        design={'model':spec,'source_run':original,'source_current':current,'anchors':baseline+controls,
            'fixed_controls':controls,'dataset':old['datasets']['confirmation'],
            'input_sha256':{str(p.resolve()):sha(p)for p in inputs},
            'configuration_sha256':{p.name:sha(p)for p in(run/'configs').glob('*.json')},
            'scope':'Fixed uniform phase controls, no optimizer/quality-budget claim.'}
        write(run/'configs/sweep_design.json',design)
        write(run/'results/sweep_design_checks.json',{'verified':True,'maps':7,'new_nll_jobs':2,'new_generation_jobs':6})
        p=run/'code/sweep_resources.py';p.write_text(p.read_text().replace("['034','035']","['038','039']"))
        p=run/'code/sweep_driver.py';text=p.read_text()
        text=text.replace("        for stage in ['evaluation','generation']:",
            "        while not(RUN.parent/'037/results/final_audit.json').exists():\n            state('baseline_replay_audit');time.sleep(10)\n        replay=read(RUN.parent/'037/results/final_audit.json')\n        assert replay['all_token_sequences_exact'] and replay['all_nll_within_original_gate'], 'Resolve baseline replay differences first'\n        seed_cached_references()\n        for stage in ['evaluation','generation']:")
        insertion='''\ndef seed_cached_references():
    import shutil
    source_run=read(RUN/'configs/sweep_design.json')['source_current']
    cache=RUN.parents[1]/'night161/checkpoints'/source_run
    with(RUN.parents[1]/'night151/export.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        for pid in ['uniform_dense_bf16','uniform_int8','uniform_sq_int8','uniform_fp8','phase_sq_int8_w4a16_ours']:
            source=cache/pid;target=TEMP/'checkpoints'/pid
            assert(source/'model.safetensors').exists(),str(source)
            assert read(source/'config.json')['quantization_config']['method_map']==read(RUN/f'configs/policies/{pid}.json')['method_map']
            if(target/'model.safetensors').exists():continue
            target.mkdir(parents=True,exist_ok=True)
            for path in source.iterdir():
                if path.is_file()and path.name!='model.safetensors':shutil.copy2(path,target/path.name)
            os.link(source/'model.safetensors',target/'model.safetensors')
            write(RUN/f'results/checkpoint_reuse/{pid}.json',{'source':str(source),'target':str(target),'exact_hardlink':True,'physical_copy_bytes':0,'effective_map_equal':True})

'''
        text=text.replace('\ndef main():',insertion+'\ndef main():');p.write_text(text)
        p=run/'code/sweep_report.py';text=p.read_text()
        text=text.replace("save('sweep_results.csv',rows)","save('sweep_results.csv',rows or all_rows)")
        text=text.replace('完整空间P+D预算与prefill权重','统一8bit-prefill/W4A16-decode控制')
        text=text.replace('沿用本模型八候选精度代理与实测LUT，去除旧截距并以BF16=0锚定；128模块独立选择64种阶段组合。全部六标签在本轮GPU输出前冻结，同速解再最小化预测损失。',
            '固定两种既有全局阶段map，与已有SQ-prefill/W4-decode及uniform点比较；所有128模块的prefill分别为plain INT8或FP8，decode均W4A16。不是DASO求解器新选点，也未改精度模型。')
        text=text.replace('q是代理/加权偏好预算，不是已验证的真实ΔNLL保证。全部点保留，不根据最终生成分数重新训练或声称通用最优。原Prefill-only及原15行主表不改，本轮没有新增MC任务。',
            '本轮无代理预算/权重调参；NLL只作质量诊断，无strictε或优化器获胜声明。全部固定控制保留，原Prefill-only/原主表不改，无新增MC。')
        text=text.replace("    for r in rows:\n        lines.append", "    for r in rows:\n        lines.append")
        text=text.replace("    lines+=['','## 自动分配'", "    lines += ['', '| 固定配置 | 实测ΔNLL | 保护值 |', '| --- | ---: | ---: |']\n    for pid in registry:lines.append(f'| {pid} | {nll[pid][\"delta_nll\"]:.5f} | {nll[pid][\"guarded_delta_nll\"]:.5f} |')\n    lines+=['','## 固定分配'")
        text=text.replace('for p in points:\n        c=counts',"for p in [{'label':pid,'policy':pid}for pid in registry]:\n        c=counts")
        p.write_text(text)
        (run/'README.md').write_text(f'# {spec["model_id"]}统一阶段控制\n\n等待037基线重放核查后，补plain INT8/FP8 prefill＋W4A16 decode两固定控制，与原SQ对应点及BF16/INT8/SQ/FP8比较。2新NLL、6新完整生成、7map统一三轮速度；不标为DASO求解器结果。原数据与框架不改。\n\n[计划](../../../../dev/plans/164_llama_uniform_phase_controls_plan.md) · [固定配置](configs/sweep_design.json) · [状态](results/sweep_worker.json)\n')
        print('Prepared',rid,spec['model_id'],'7 maps,2 newNLL,6newGEN',flush=True)


if __name__=='__main__':main()
