"""Audit and seal the new study while preserving every earlier archive."""
import hashlib
import json
import os
import shutil
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()


def main():
    assert not (RUN/'results/manifest.json').exists()
    report=read(RUN/'results/report_complete.json');assert report['complete'] and report['model_processes']==28
    assert read(RUN/'results/worker.json')['status']=='complete'
    assert report['same_output_all_passed'],'cross-process or node output drift needs investigation'
    design=read(RUN/'configs/design.json');selection=read(RUN/'configs/sla_selected.json')
    assert selection['frozen_before_test'] and selection['design_sha256']==digest(RUN/'configs/design.json')
    old=read(RUN/'reference/protected_manifests.json')
    for rid,expected in old.items():assert digest(RUN.parent/rid/'results/manifest.json')==expected
    syntax=0
    for p in (RUN/'code').rglob('*.py'):compile(p.read_text(),str(p),'exec');syntax+=1
    commands=0;failed=[];versions=set()
    for p in sorted((RUN/'logs').glob('*.command.json')):
        r=read(p);commands+=1;version=r['code_version'];versions.add(version)
        for name,expected in r['code_hashes'].items():assert digest(RUN/'code_versions'/version/name)==expected
        assert r['command'][0]=='numactl' and '--membind='+str(r['node']) in r['command']
        done=read(p.with_name(p.name.replace('.command.json','.completion.json')))
        if done['returncode']:failed.append({'command_record':str(p.relative_to(RUN)),**done})
        if '--round' in r['command'] and int(r['command'][r['command'].index('--round')+1])>=2:
            assert int(p.name.removesuffix('.command.json').rsplit('_',1)[1])>selection['time_ns']
    assert len(failed)==1 and 'copy_0_node0' in failed[0]['command_record']
    expected={'_SYNC_COPY_ENABLED':False,'_RELEASE_PREFILL_ENABLED':True,'_GC_EVERY_APPLY_ENABLED':True,
        '_GC_DECODE_ENABLED':False,'_WAIT_ONCE_ENABLED':True}
    for p in (RUN/'results/speed').glob('*/*/*.json'):assert read(p)['runtime_settings']==expected
    audit={'passed':True,'model_processes':28,'case_records':56,'timed_iterations':448,'copy_processes':4,
        'command_records_verified':commands,'code_versions_verified':len(versions),'syntax_files':syntax,
        'historical_failures_preserved':failed,'early_first_token_fields_excluded':report['early_first_token_observer_excluded'],
        'same_output_checks':report['same_output_checks'],'same_output_checks_passed':True,
        'selection_frozen_before_new_latency_test':True,'new_quality_labels':0,'protected_manifests_unchanged':old,
        'runtime_settings_verified':expected,'scope':'Empirical latency reproducibility and validation; no claim of strict future SLA/quality guarantee.'}
    write(RUN/'results/final_audit.json',audit);entries=[]
    for p in sorted(RUN.rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
        relative=str(p.relative_to(RUN))
        if relative.startswith('extensions/') or relative=='results/manifest.json':continue
        entries.append({'path':relative,'bytes':p.stat().st_size,'sha256':digest(p)})
    write(RUN/'results/manifest.json',{'files':entries});manifest=digest(RUN/'results/manifest.json')
    packages=[]
    for reviewer in [1,2,3,5]:
        target=RUN.parents[1]/'responses'/f'reviewer{reviewer}/reproducibility_run012';target.mkdir(parents=True,exist_ok=True);copies={}
        files=list(RUN.glob('*.md'))+list((RUN/'results').glob('*.csv'))
        files += [RUN/p for p in ['results/final_audit.json','results/report_complete.json','results/sla_audit.json',
            'results/output_equivalence.json','results/observer_correction.json','configs/design.json','configs/sla_selected.json']]
        files += list((RUN/'results/figures').glob('*'))
        for source in files:
            relative='RUN_README.md' if source==RUN/'README.md' else str(source.relative_to(RUN))
            dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            assert digest(dest)==digest(source);copies[relative]=digest(dest)
        links={}
        for name,source in [('code',RUN/'code'),('data',RUN/'data'),('reference',RUN/'reference'),('raw_results',RUN/'results'),('code_versions',RUN/'code_versions')]:
            relative=os.path.relpath(source,target);link=target/name
            if not link.is_symlink():link.symlink_to(relative)
            assert link.exists() and link.resolve()==source.resolve();links[name]=relative
        (target/'README.md').write_text('# 排名与约束可靠性复核（run012）\n\n[完整报告](../../../runs/012/REPORT.md) · [CSV](results/process_summary.csv) · [SLA审计](results/sla_audit.json) · [原始结果](raw_results)。\n\n原001–011保留；无新增质量标签，NUMA/进程重复/时延holdout分别报告。\n')
        write(target/'source.json',{'run':'012','manifest_sha256':manifest,'copies':copies,'links':links});packages.append(str(target))
    for entry in entries:assert digest(RUN/entry['path'])==entry['sha256']
    print(json.dumps({'passed':True,'sealed_files':len(entries),'manifest_sha256':manifest,'reviewer_packages':packages}))


if __name__=='__main__':main()
