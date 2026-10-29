"""Seal results and prove that prior evidence and selection stayed unchanged."""
import hashlib
import json
from pathlib import Path

from common import RUN, REPO, VLLM, write_json
from launch import read


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(base, expected):
    path = base / 'results/manifest.json'
    assert sha(path) == expected, str(path)
    entries = read(path)['files']
    entries = entries.items() if isinstance(entries,dict) else ((x['path'],x) for x in entries)
    count = 0
    for name,entry in entries:
        path = base/name
        if 'symlink' in entry:
            assert path.is_symlink() and str(path.resolve()) == entry['symlink'],str(path)
        if 'bytes' in entry:
            assert path.stat().st_size == entry['bytes'],str(path)
        if 'sha256' in entry:
            assert sha(path) == entry['sha256'],str(path)
        count += 1
    return count


def main():
    assert read(RUN/'results/report_complete.json')['full_tasks_complete']
    assert read(RUN/'results/training_reference_compatible.json')['passed']
    assert read(RUN/'results/nvfp4_decode_finalize_gate.json')['passed']
    for stage in ['training','validation','confirmation','full_quality','formal','calibration']:
        assert read(RUN/f'results/{stage}_complete.json')['complete'],stage
    pool = read(RUN/'configs/allocation_pool.json')
    selection = read(RUN/'configs/selected_policies.json')
    assert pool['design_sha256'] == sha(RUN/'configs/frozen_design.json')
    assert selection['pool_sha256'] == sha(RUN/'configs/allocation_pool.json')
    assert selection['design_sha256'] == pool['design_sha256']
    design = read(RUN/'configs/frozen_design.json')
    for scenario,digest in design['training_manifest_sha256'].items():
        assert sha(RUN/f'configs/calibration_{scenario}.json') == digest
        for entry in read(RUN/f'configs/calibration_{scenario}.json'):
            assert sha(RUN/f'configs/policies/{entry["policy_id"]}.json') == entry['policy_sha256']
    assert sha(RUN/'configs/training_jobs.json') == design['training_jobs_sha256']
    for split,spec in design['datasets'].items():
        assert not set(spec['starts']) & set(spec['excluded_old_starts'])
        assert sha(Path(spec['file'])) == spec['sha256']
        for path in (RUN/f'results/{split}').glob('*/*.json'):
            result = read(path)
            assert result['data_sha256'] == spec['sha256'] and len(result['blocks']) == spec['blocks'],str(path)
            assert result['quality_score_batch'] == 8,str(path)
            expected = 2047 if path.stem == 'prefill_only' else 64
            assert all(b['token_count'] == expected for b in result['blocks']),str(path)
    for pid,spec in pool['policies'].items():
        mapping = read(RUN/f'configs/policies/{pid}.json')['method_map']
        digest = hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
        assert digest == spec['signature']
    protected = read(RUN.parent/'015/reference/protected_manifests.json')
    protected['015'] = read(RUN/'reference/protected_run015_manifest.json')['sha256']
    count = sum(verify_manifest(RUN.parent/rid,digest) for rid,digest in protected.items())
    binary = VLLM/'vllm/vllm/_C.abi3.so'
    assert sha(binary) == 'a6a469b427f36a61582e1fa99d8f83cf0fdd230e836e5eae7fe01514cd65fc2c'
    # Final state is written before hashing; the driver repeats these same bytes.
    write_json(RUN/'results/worker.json',{'status':'complete','stage':'audited'})
    active = set()
    failed = []
    for path in (RUN/'logs').glob('*.command.json'):
        done = path.with_name(path.name.replace('.command.json','.completion.json'))
        if not done.exists():
            active.add(path.with_name(path.name.replace('.command.json','.log')))
        elif read(done)['returncode']:
            failed.append({'command':str(path.relative_to(RUN)),**read(done)})
    write_json(RUN/'results/retained_failures.json',failed)
    files = {}
    for path in sorted(RUN.rglob('*')):
        name = str(path.relative_to(RUN))
        if name in ['results/manifest.json','results/final_audit.json']:
            continue
        if path.is_symlink():
            files[name] = {'symlink':str(path.resolve())}
        elif path.is_dir():
            continue
        elif path in active or path.name == 'driver.log':
            files[name] = {'mutable_orchestration_log':True}
        else:
            files[name] = {'sha256':sha(path),'bytes':path.stat().st_size}
    write_json(RUN/'results/manifest.json',{'files':files,
        'scope':'All code/configs/raw results/reports/failures. Active controller logs may append their final exit record.'})
    write_json(RUN/'results/final_audit.json',{'complete':True,'verified':True,
        'protected_runs':len(protected),'protected_files_verified':count,
        'allocation_labels':len(pool['points']),'unique_policies':len(pool['policies']),
        'archived_files':len(files),'retained_failed_commands':len(failed),
        'manifest_sha256':sha(RUN/'results/manifest.json')})
    print('Run016 sealed; prior files verified',count,flush=True)


if __name__ == '__main__':
    main()
