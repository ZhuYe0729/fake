"""Verify bounded overnight experiments, then seal and package local evidence."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

HOME_RUN=Path(__file__).resolve().parents[1]
RUNS=HOME_RUN.parent
REVIEWERS={'004':[1,3,4,5],'005':[1,5],'006':[1,2],'007':[4],
    '008':[4,5],'009':[1,2,3],'010':[1,3,4],'011':[1,2,3,5]}


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def csv_count(path):
    with path.open(newline='') as f:return sum(1 for _ in csv.DictReader(f))


def status(root,name):assert read(root/f'results/{name}.json')['status']=='complete',name


def audit(rid):
    root=RUNS/rid;proof=read(HOME_RUN/'results/protected_integrity.json');assert proof['passed']
    assert not (root/'results/manifest.json').exists(),f'{rid} already sealed'
    assert sum(proof['protected_archive_file_counts'].values())==46527
    for old,expected in proof['protected_manifest_sha256'].items():assert digest(RUNS/old/'results/manifest.json')==expected
    assert (root/'REPORT.md').exists()
    syntax=0
    for path in (root/'code').rglob('*.py'):
        compile(path.read_text(),str(path),'exec');syntax+=1
    commands=[];failed=[];snapshots=set()
    for path in sorted((root/'logs').glob('*.command.json')):
        command=read(path);version=command['code_version'];snapshots.add(version)
        for name,expected in command['code_hashes'].items():assert digest(root/'code_versions'/version/name)==expected,(path,name)
        completion=read(path.with_name(path.name.replace('.command.json','.completion.json')))
        record={'command_record':str(path.relative_to(root)),'code_version':version,**completion}
        commands.append(record)
        if completion['returncode']!=0:failed.append(record)
    assert len(failed)=={'004':2,'006':1,'011':1}.get(rid,0),(rid,failed)
    details={}
    if rid=='004':
        for name in ['runtime_worker','profile_worker']:status(root,name)
        r=read(root/'results/report_complete.json');assert r['runtime_rows']==36 and r['profiles']==12 and r['semantic_equivalence']
        assert csv_count(root/'results/runtime_summary.csv')==36
        for name,count in [('interfaces',98),('large_shapes',112)]:
            j=read(root/f'results/{name}_complete.json');assert j['passed'] and j['rows']==count
        assert read(root/'results/runtime_semantics_audit.json')
        details={'runtime_rows':36,'profiles':12,'interfaces':98,'real_large_shape_measurements':112,
            'recovered_historical_failures':['results/large_shapes_recovery.json','../006/results/full_quality_worker.json']}
    elif rid=='005':
        status(root,'pipeline');selection=read(root/'configs/selected_policies.json');pool=read(root/'configs/candidate_pool.json')
        assert selection['frozen_before_test'] and pool['frozen_before_validation']
        for path,key in [('configs/candidate_pool.json','pool_sha256'),('configs/design.json','rules_sha256'),('results/validation_selection_audit.json','selection_audit_sha256')]:
            assert digest(root/path)==selection[key]
        assert digest(root/'configs/selected_policies.json')=='fc5987c6a622a50b5c0a0b2fc4fe36b770215576a5c87838da51f948c053ac21'
        selection_time=(root/'configs/selected_policies.json').stat().st_mtime_ns
        for path in (root/'logs').glob('*.command.json'):
            argv=read(path)['command']
            if '--mode' in argv and argv[argv.index('--mode')+1] in ['test','test_speed']:
                assert int(path.name.removesuffix('.command.json').rsplit('_',1)[1])>selection_time
        for pid,info in pool['policies'].items():
            policy=read(root/f'configs/policies/{pid}.json')
            assert hashlib.sha256(json.dumps(policy['method_map'],sort_keys=True).encode()).hexdigest()==info['signature']
        design=read(root/'configs/design.json')
        for split,info in design['data'].items():
            assert digest(root/f'data/{split}_64_2112.pt')==info['file_sha256']
            starts=sorted(info['starts']);assert all(b-a>=info['length'] for a,b in zip(starts,starts[1:]))
        assert len(pool['policies'])==38 and csv_count(root/'results/summary_final.csv')==42
        j=read(root/'results/quality_sla_audit.json');verified=[r for r in j['quality'] if r['family']=='DASO plus validation']
        assert sum(r['points'] for r in verified)==8 and sum(r['violations'] for r in verified)==0
        for name in ['exact_solver_audit','surrogate_audit']:assert read(root/f'results/{name}.json')['passed']
        details={'pool_policies':38,'final_comparison_rows':42,'selected_unique_policies':len(selection['policies']),
            'independent_test_quality_points':8,'test_quality_violations':0,
            'strict_sla_claim':False,'quality_sla_results':j}
    elif rid=='006':
        for name in ['worker_speed','worker_quality','full_quality_worker','extension_worker','repeat_worker']:status(root,name)
        r=read(root/'results/report_complete.json')
        assert r['speed_cases']==21 and r['actual_workload_quality_files']==18
        assert r['extended_speed_cases_with_brackets']==27 and r['extended_comparisons']==24 and r['extended_quality_new_files']==6
        assert len(list((root/'results/serving_extended').glob('*/*.json')))==27
        assert len(list((root/'results/full_quality_extended').glob('*/*.json')))==6
        assert len(list((root/'results/serving_repeated').glob('*/*.json')))==27
        assert r['counterbalanced_repeat_speed_cases']==27 and csv_count(root/'results/serving_counterbalanced_summary.csv')==24
        assert csv_count(root/'results/serving_extended_summary.csv')==24
        design=read(root/'configs/baseline_extension_design.json')
        assert digest(root/'configs/design.json')==design['old_design_sha256']
        assert digest(RUNS/'009/configs/candidate_pool.json')==design['hawq_pool_sha256']
        q=read(root/'configs/full_quality_design.json');assert digest(root/'data/validation_16_4352.pt')==q['data_sha256']
        assert all(read(p)['passed'] for p in (root/'results/routing_audits').glob('*.json'))
        details={'initial_speed_cases':21,'extension_speed_cases':27,'repeat_speed_cases':27,'final_comparisons':24,
            'actual_workload_quality_files':24,'full_qa_windows':384,'strict_budget_guarantee':False,
            'scope':'Actual vLLM mixed-token chunked scheduler; dual resident experiment; closed-loop offline bursts, not open-loop production serving.'}
    elif rid=='007':
        for name in ['preparation_worker','evaluation_worker']:status(root,name)
        assert read(root/'results/pruning_formula_audit.json')['passed'] and csv_count(root/'results/summary_final.csv')==8
        d=read(root/'configs/design.json')['test_data'];assert digest(root/'data/test_64_2112.pt')==d['sha256']
        details={'methods':4,'same_kernel_rows':8,'new_structured_modules_checked':504,'full_downstream':'run010'}
    elif rid=='008':
        fig=read(root/'results/figure2_audit.json');assert fig['passed'] and fig['output_pdf_font_count']==0 and fig['visual_inspection']
        assert digest(Path(fig['source']))==fig['source_sha256']
        for name in ['NAS_CONTEXT.md','IDENTICAL_INPUTS.md']:assert (root/name).exists()
        details={'figure2_zero_fonts':True,'figure2_visual_inspection':True,'protected_integrity':proof}
        write(root/'results/report_complete.json',{'complete':True,**details})
    elif rid=='009':
        status(root,'pipeline');r=read(root/'results/report_complete.json')
        assert r['points']==35 and r['unique_policies']==16 and r['all_transforms_retained']
        assert csv_count(root/'results/summary_final.csv')==35
        pool=read(root/'configs/candidate_pool.json');assert pool['frozen_before_quality']
        for pid,info in pool['policies'].items():
            p=read(root/f'configs/policies/{pid}.json')
            assert hashlib.sha256(json.dumps(p['method_map'],sort_keys=True).encode()).hexdigest()==info['signature']
        details={'all_predeclared_variants':35,'unique_maps':16,'speed_gpu':0,'full_cnn_qat_reproduction':False}
    elif rid=='010':
        for name in ['worker_tasks_3','worker_tasks_5','worker_generation_0','worker_generation_6']:status(root,name)
        details=read(root/'results/task_audit.json');assert details['passed'] and details['all_preregistered_jobs_complete']
        assert csv_count(root/'results/likelihood_summary.csv')==30 and csv_count(root/'results/generation_summary.csv')==8
        for p in (root/'results/tasks').glob('*/*/full/result.json'):
            r=read(p);assert r['limit'] is None and r['num_fewshot']==0
            assert all(n['effective']==n['original'] for n in r['raw_lm_eval']['n-samples'].values())
        for r in read(root/'results/inherited_tasks.json')['records']:assert digest(root/r['target'])==r['sha256']
    elif rid=='011':
        status(root,'worker_speed');status(root,'worker_quiet');r=read(root/'results/report_complete.json')
        assert r['comparison_points']==43 and r['unique_maps']==19 and r['new_quality_labels']==0
        assert csv_count(root/'results/summary_final.csv')==43
        assert len(list((root/'results/calibration_speed').glob('*/*.json')))==25
        assert len(list((root/'results/calibration_speed_quiet').glob('*/*.json')))==25 and r['quiet_host_primary']
        pool=read(root/'configs/candidate_pool.json');assert pool['frozen_before_new_speed']
        assert digest(root/'configs/design.json')==pool['design_sha256']
        assert digest(root/'configs/candidate_pool.json')==read(root/'configs/quiet_campaign.json')['pool_sha256']
        for record in read(root/'reference/inherited_quality.json')['records']:
            assert digest(root/record['target'])==digest(Path(record['source']))==record['sha256']
        details=r
    report=read(root/'results/report_complete.json');assert report.get('complete',report.get('passed'))
    final={'passed':True,'run':rid,'syntax_checked_files':syntax,'command_records_verified':len(commands),
        'code_snapshots_verified':len(snapshots),'historical_failed_attempts_preserved':failed,
        'failed_attempts_resolved_in_final_outputs':True,'checks':details,
        'protected_integrity_proof_sha256':digest(HOME_RUN/'results/protected_integrity.json'),
        'scope':'Report-specific numerical checks plus archived command/source provenance. Negative scientific outcomes remain valid results; recovered diagnostic attempts are retained.'}
    write(root/'results/final_audit.json',final)
    print(json.dumps({'run':rid,'audit':'passed','commands':len(commands),'snapshots':len(snapshots),'historical_failures':len(failed)}),flush=True)


def package(rid):
    root=RUNS/rid;assert read(root/'results/final_audit.json')['passed']
    path=root/'results/manifest.json';entries=[]
    if path.exists():
        entries=read(path)['files']
        for entry in entries:assert digest(root/entry['path'])==entry['sha256']
    else:
        for source in sorted(root.rglob('*')):
            if not source.is_file() or source.is_symlink() or '__pycache__' in source.parts:continue
            relative=str(source.relative_to(root))
            if relative.startswith('extensions/') or relative=='results/manifest.json':continue
            entries.append({'path':relative,'bytes':source.stat().st_size,'sha256':digest(source)})
        write(path,{'files':entries})
    manifest_hash=digest(path)
    copies=[p for p in root.glob('*.md')]
    copies += list((root/'results').glob('*.csv'))
    copies += [p for p in (root/'results').glob('*.json') if p.name not in ['manifest.json','progress.json'] and p.stat().st_size<2*1024**2]
    copies += list((root/'configs').glob('*.json'))
    if rid=='008':copies += [root/'results'/n for n in ['figure2_outlined.pdf','figure2_outlined.svg','figure2_outlined.png']]
    for reviewer in REVIEWERS[rid]:
        target=RUNS.parent/'responses'/f'reviewer{reviewer}'/f'overnight_run{rid}';target.mkdir(parents=True,exist_ok=True);hashes={}
        for source in copies:
            relative='RUN_README.md' if source==root/'README.md' else str(source.relative_to(root));dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest);assert digest(dest)==digest(source);hashes[relative]=digest(dest)
        for source in (root/'results/figures').glob('*'):
            if source.is_file():
                relative=str(source.relative_to(root));dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest);hashes[relative]=digest(dest)
        links={}
        for name in ['code','data','reference','code_versions','raw_results','all_configs']:
            original={'raw_results':'results','all_configs':'configs'}.get(name,name);source=root/original
            if not source.exists():continue
            link=target/name;relative=os.path.relpath(source,link.parent)
            if not link.is_symlink():link.symlink_to(relative)
            assert link.resolve()==source.resolve();links[name]=relative
        (target/'README.md').write_text(f'# 人工reviewer补充证据：run{rid}\n\n'
            f'[完整报告及跨实验引用](../../../runs/{rid}/REPORT.md) · [报告副本](REPORT.md) · [审计](results/final_audit.json) · [完整原始结果](raw_results) · [代码](code) · [全部配置](all_configs)。\n\n'
            '最终CSV/报告/图及关键审计为本目录副本，code/data/raw_results等链接指向独立编号实验；所有失败和不利结果保留。source.json记录来源manifest与副本SHA。\n')
        write(target/'source.json',{'run':rid,'manifest_sha256':manifest_hash,'copies':hashes,'links':links})
    for entry in entries:assert digest(root/entry['path'])==entry['sha256']
    print(json.dumps({'run':rid,'sealed_files':len(entries),'manifest_sha256':manifest_hash,'reviewer_packages':len(REVIEWERS[rid])}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','package']);parser.add_argument('--runs',default=','.join(REVIEWERS));args=parser.parse_args()
    for rid in args.runs.split(','):(audit if args.mode=='audit' else package)(rid)


if __name__=='__main__':main()
