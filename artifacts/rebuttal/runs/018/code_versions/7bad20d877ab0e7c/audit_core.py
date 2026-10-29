"""Seal the completed core table, explicitly retaining unmeasured supplements."""
import hashlib
import json
import math
from pathlib import Path

from common import RUN, SPEC
from launch import read, write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = read(RUN / 'results/report_complete.json')
    assert read(RUN / 'configs/primary_policies.json')['reporting_scope_sha256'] == sha(RUN / 'configs/core_reporting_scope.json')
    assert report['full_tasks_complete'] and report['main_rows'] == 15
    for stage in ['training','validation','calibration','confirmation','formal','full_quality']:
        assert read(RUN / f'results/{stage}_complete.json')['complete']
    assert read(RUN / 'results/label_cache_decision.json')['complete']
    frozen = read(RUN / 'configs/frozen_design.json')
    assert frozen['generation_protocol_sha256'] == sha(RUN / 'configs/generation_protocol.json')
    for task, spec in read(RUN / 'configs/generation_protocol.json')['datasets'].items():
        assert sha(RUN / f'data/generation_{task}.jsonl') == spec['sha256']
    for scene,digest in frozen['training_manifest_sha256'].items():
        assert sha(RUN / f'configs/calibration_{scene}.json') == digest
        for entry in read(RUN / f'configs/calibration_{scene}.json'):
            pid = entry['policy_id']
            assert sha(RUN / f'configs/policies/{pid}.json') == entry['policy_sha256']
            result = read(RUN / f'results/training/{scene}/{pid}.json')
            assert len(result['blocks']) == 100 and math.isfinite(result['avg_nll'])
            if 'source_sha256' in entry:
                assert sha(RUN / f'results/training/{scene}/{pid}.json') == entry['source_sha256']
    assert sha(RUN / 'configs/training_jobs.json') == frozen['training_jobs_sha256']
    pool = read(RUN / 'configs/allocation_pool.json')
    assert read(RUN / 'results/source_identity_gate.json')['passed']
    assert pool['proposal_rules_sha256'] == sha(RUN / 'configs/proposal_rules.json')
    assert pool['original_main_sha256'] == sha(RUN / 'configs/original_main_policies.json')
    assert pool['profile_sha256'] == sha(RUN / 'reference/graph_action_profile.csv')
    checks = list((RUN / 'results/kernel_checks').glob('*.json'))
    assert len(checks) == 16 and all(read(p)['passed'] for p in checks)
    assert pool['design_sha256'] == sha(RUN / 'configs/frozen_design.json')
    selected = read(RUN / 'configs/selected_policies.json')
    assert selected['pool_sha256'] == sha(RUN / 'configs/allocation_pool.json')
    assert selected['design_sha256'] == pool['design_sha256']
    for split in ['validation','confirmation']:
        dataset = frozen['datasets'][split]
        assert sha(Path(dataset['file'])) == dataset['sha256']
        for pid, spec in pool['policies'].items():
            for scene in spec['scenarios']:
                result = read(RUN / f'results/{split}/{pid}/{scene}.json')
                assert result['data_sha256'] == dataset['sha256']
                assert len(result['blocks']) == dataset['blocks'] and result['quality_score_batch'] == 8
                assert all(b['token_count'] == (2047 if scene=='prefill_only' else 64) for b in result['blocks'])
    protected = read(RUN / 'reference/protected_run016_manifest.json')
    assert sha(RUN.parent / '016/results/manifest.json') == protected['sha256']
    for original in read(RUN / 'configs/original_main_policies.json'):
        assert sha(Path(original['source'])) == original['source_sha256']
    binary = Path('/root/wja/project/my/cospaq/test/vllm/vllm/vllm/_C.abi3.so')
    assert sha(binary) == 'a6a469b427f36a61582e1fa99d8f83cf0fdd230e836e5eae7fe01514cd65fc2c'
    # The driver repeats these exact bytes after the audit subprocess returns.
    write(RUN / 'results/worker.json', {'status':'complete','stage':'core_table_complete',
        'scope':'Core main table and central controls finished; supplemental speed points remain explicitly separate.'})
    active, failures = set(), []
    for path in (RUN / 'logs').glob('*.command.json'):
        done = path.with_name(path.name.replace('.command.json','.completion.json'))
        if not done.exists():
            active.add(path.with_name(path.name.replace('.command.json','.log')))
        elif read(done)['returncode']:
            failures.append({'command':str(path.relative_to(RUN)),**read(done)})
    write(RUN / 'results/retained_failures.json', failures)
    files = {}
    for path in sorted(RUN.rglob('*')):
        name = str(path.relative_to(RUN))
        if name in ['results/manifest.json','results/final_audit.json']:
            continue
        if path.is_symlink():
            files[name] = {'symlink':str(path.resolve())}
        elif path.is_dir():
            continue
        elif path in active or path.name.startswith('driver'):
            files[name] = {'mutable_orchestration_log':True}
        else:
            files[name] = {'sha256':sha(path),'bytes':path.stat().st_size}
    write(RUN / 'results/manifest.json',{'files':files,
        'scope':'Core main tables and central controls; remaining allocated speed points explicitly pending, not counted as measured.'})
    write(RUN / 'results/final_audit.json',{'complete':True,'verified':True,'scope':'core_table',
        'main_rows':15,'primary_policies':report['primary_policies'],
        'supplemental_speed_pending':report['supplemental_speed_pending'],
        'archived_files':len(files),'retained_failed_commands':len(failures),
        'manifest_sha256':sha(RUN / 'results/manifest.json')})
    print('Sealed core table',SPEC['model_id'],'files',len(files),flush=True)


if __name__ == '__main__':
    main()
