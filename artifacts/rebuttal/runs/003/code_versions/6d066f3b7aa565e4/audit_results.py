"""Validate frozen coverage, tokens, quality provenance and inherited archives."""
import ast
import hashlib
import json
from pathlib import Path

from common import RUN, write_json


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def main():
    progress=json.loads((RUN/'results/progress.json').read_text())
    assert progress['complete'] and not progress['missing']
    assert all(w['status']=='complete' for w in progress['workers'].values()),progress['workers']
    design=json.loads((RUN/'configs/design.json').read_text())
    final_gpu=json.loads((RUN/'configs/measurement_card_change.json').read_text())['final_speed_gpu']
    frozen=json.loads((RUN/'configs/evaluation_points.json').read_text())
    assert frozen['design_sha256']==digest(RUN/'configs/design.json')
    assert len(design['workloads'])==16 and len(frozen['points'])==267 and len(frozen['policies'])==47
    assert len({w['id'] for w in design['workloads']})==16
    for pid,job in frozen['policies'].items():
        p=json.loads((RUN/f'configs/policies/{pid}.json').read_text())
        assert len(p['method_map'])==144
        assert hashlib.sha256(json.dumps(p['method_map'],sort_keys=True).encode()).hexdigest()==job['signature']
        for wid in job['workloads']:
            path=RUN/f'results/speed/{pid}/summary_{wid}.json';assert path.exists()
            result=json.loads(path.read_text());assert result['gpu']==str(final_gpu) and result['kv_bytes']==6*1024**3
    iterations=0;speed_groups=0
    for path in (RUN/'results/speed').glob('*/*.json'):
        if path.name.startswith('summary_'):continue
        result=json.loads(path.read_text());records=result['iterations']
        assert result['gpu']==str(final_gpu) and result['reservation_released']
        assert len(records)==7 and sum(not r['warmup'] for r in records)==5
        for r in records:
            assert r['requests']==result['batch']
            assert r['generated_tokens']==result['batch']*result['output']
            assert r['elapsed_ms']>0
        iterations+=len(records);speed_groups+=1
    quality_files=0;scored_tokens=0;scored_blocks=0
    for path in (RUN/'results/quality').glob('*/*.json'):
        q=json.loads(path.read_text());blocks=q['blocks'];assert len(blocks)==100
        assert [b['block'] for b in blocks]==list(range(100))
        w=q['workload'];count=w['input']-1 if w['scenario']=='prefill_only' else w['output']
        assert q['token_count']==count*100 and all(b['token_count']==count for b in blocks)
        assert abs(sum(b['nll'] for b in blocks)-q['total_nll'])<1e-7
        if w['scenario']=='prefill_decode':assert q['phase_events']['enter_decode'] and q['phase_events']['apply_decode']
        quality_files+=1;scored_tokens+=q['token_count'];scored_blocks+=100
    numerical=json.loads((RUN/'results/int8_fusion_audit.json').read_text());assert numerical['passed']
    snapshots={}
    for command in (RUN/'logs').glob('*.command.json'):
        record=json.loads(command.read_text())
        for name,expected in record['code_hashes'].items():
            path=RUN/'code_versions'/record['code_version']/name
            if path not in snapshots:snapshots[path]=digest(path)
            assert snapshots[path]==expected,(str(command),name)
    inherited={};sources=json.loads((RUN/'reference/sources.json').read_text())
    for rid in ['001','002']:
        old=RUN.parent/rid;manifest=old/'results/manifest.json'
        assert digest(manifest)==sources['inherited_manifests'][rid]
        entries=json.loads(manifest.read_text())['files']
        for entry in entries:
            assert digest(old/entry['path'])==entry['sha256'],(rid,entry['path'])
        inherited[rid]=len(entries)
    for scenario in ['prefill_only','prefill_decode']:
        assert digest(RUN/f'reference/quality_model_{scenario}.json')==digest(RUN.parent/f'002/results/daso8/{scenario}/quality_model.json')
    runtime_sources=0
    for relative in ['001/reference/runtime_sources/index.json','002/reference/fp8_runtime_sources/index.json']:
        for record in json.loads((RUN.parent/relative).read_text())['records']:
            assert digest(Path(record['source']))==record['sha256'],record['source']
            runtime_sources+=1
    environment=json.loads((RUN.parent/'002/configs/environment.json').read_text())
    for source,expected in environment['compiled_vllm_ops'].items():assert digest(Path(source))==expected['sha256'],source
    for name in ['eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py']:
        assert digest(RUN/'code'/name)==digest(RUN.parent/'002/code'/name),name
    def function(path,name):
        return next(ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name==name)
    assert function(RUN/'code/common.py','quantize_int8')==function(RUN.parent/'002/code/common.py','quantize_int8')
    failed_attempts=[str(path.relative_to(RUN)) for path in (RUN/'logs').glob('*.completion.json')
                     if json.loads(path.read_text())['returncode']!=0]
    replay=json.loads((RUN/'results/allocation_replay_audit.json').read_text());assert replay['passed']
    metadata=json.loads((RUN/'data/test_samples.json').read_text())
    assert digest(RUN/'data/test_100_4352.pt')==metadata['sha256']
    result={'passed':True,'workloads':16,'comparison_points':267,'unique_policies':47,
        'unique_policy_workloads':progress['speed_measurements'],'speed_groups':speed_groups,'speed_iterations':iterations,
        'quality_files':quality_files,'scored_blocks':scored_blocks,'scored_tokens':scored_tokens,
        'int8_numeric_cases':len(numerical['cases']),'inherited_files_verified':inherited,
        'code_snapshot_files_verified':len(snapshots),
        'current_runtime_sources_unchanged':runtime_sources,'compiled_vllm_ops_unchanged':True,
        'frozen_quality_models_unchanged':True,'ordinary_int8_weight_quantizer_unchanged':True,
        'all_final_workers_complete':True,'final_speed_gpu':final_gpu,'gpu3_partial_retained':True,
        'gpu3_partial_policy_workloads':len(list((RUN/'results/diagnostics/speed_gpu3_partial').glob('*/summary_*.json'))),
        'retired_quality_workers':[k for k,v in progress['workers'].items() if v.get('retired')],
        'failed_attempts_retained':failed_attempts,'allocation_replay_identical':True,
        'scope':'actual workloads, token counts and archive consistency; not a statistical quality guarantee'}
    assert json.loads((RUN/'results/gpu0_reservation/state.json').read_text())['reserved_bytes']==0
    write_json(RUN/'results/final_audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
